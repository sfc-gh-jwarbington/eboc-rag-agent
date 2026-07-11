import streamlit as st
from snowflake.snowpark.context import get_active_session
import json
import re
import uuid
from datetime import datetime

# --- Configuration ---
AGENT_DB = "EBOC_RAG"
AGENT_SCHEMA = "PUBLIC"
AGENT_NAME = f"{AGENT_DB}.{AGENT_SCHEMA}.EBOC_AGENT"

CATEGORIES = [
    "All Specialties", "Infection", "Respiratory", "Cardiovascular", "Endocrine",
    "Perioperative", "Nervous", "Hematological", "Medications", "Behavioral",
    "Musculoskeletal", "Genitourinary", "Neonatal", "Growth", "Digestive"
]

# --- Session State Init ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "parent_message_id" not in st.session_state:
        st.session_state.parent_message_id = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"


# --- Data Loaders ---
@st.cache_data(show_spinner=False)
def load_source_urls():
    session = get_active_session()
    rows = session.sql(
        f"SELECT GUIDELINE_NAME, SOURCE_URL FROM {AGENT_DB}.{AGENT_SCHEMA}.CATEGORY_MAP"
    ).collect()
    return {r["GUIDELINE_NAME"]: r["SOURCE_URL"] for r in rows}


@st.cache_data(show_spinner=False)
def load_algorithms():
    session = get_active_session()
    rows = session.sql(
        f"SELECT GUIDELINE_NAME, CATEGORY, ALGORITHM_NAME, ALGORITHM_TYPE, "
        f"STRUCTURED_CONTENT, TEXT_SUMMARY FROM {AGENT_DB}.{AGENT_SCHEMA}.ALGORITHMS "
        f"ORDER BY CATEGORY, GUIDELINE_NAME"
    ).collect()
    return rows


@st.cache_data(ttl=300)
def load_version_alerts():
    session = get_active_session()
    rows = session.sql(
        f"SELECT RESPONSE_SUMMARY AS GUIDELINE, CREATED_AT FROM {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"WHERE LOG_TYPE = 'version_alert' AND CREATED_AT > DATEADD('day', -7, CURRENT_TIMESTAMP()) "
        f"ORDER BY CREATED_AT DESC LIMIT 10"
    ).collect()
    return rows


@st.cache_data(ttl=60)
def load_analytics():
    session = get_active_session()
    top_queries = session.sql(
        f"SELECT SPECIALTY, COUNT(*) AS cnt FROM {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"WHERE LOG_TYPE = 'query' GROUP BY SPECIALTY ORDER BY cnt DESC LIMIT 10"
    ).collect()
    feedback_summary = session.sql(
        f"SELECT FEEDBACK, COUNT(*) AS cnt FROM {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"WHERE LOG_TYPE = 'query' AND FEEDBACK IS NOT NULL GROUP BY FEEDBACK"
    ).collect()
    total_queries = session.sql(
        f"SELECT COUNT(*) AS cnt FROM {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG WHERE LOG_TYPE = 'query'"
    ).collect()
    return {"top_specialties": top_queries, "feedback": feedback_summary,
            "total": total_queries[0]["CNT"] if total_queries else 0}


# --- Citation Parsing ---
def parse_chunk_meta(text):
    if not text:
        return {}
    meta = {}
    m = re.search(r"Guideline:\s*(.*?)\s*\|", text)
    if m:
        meta["guideline"] = m.group(1).strip()
    c = re.search(r"Category:\s*([^\n|]*)", text)
    if c:
        meta["category"] = c.group(1).strip()
    s = re.search(r"Section:\s*([^\n]*)", text)
    if s:
        meta["section"] = s.group(1).strip()
    p = re.search(r"Page.*?(\d+)", text)
    if p:
        meta["page"] = int(p.group(1))
    return meta


def extract_citations(parsed, source_urls):
    citations = []
    seen = set()
    for block in parsed.get("content", []):
        if block.get("type") == "tool_result":
            for c in block.get("tool_result", {}).get("content", []) or []:
                j = c.get("json", {}) or {}
                for sr in j.get("search_results", []) or []:
                    text = sr.get("text", "")
                    meta = parse_chunk_meta(text)
                    guideline = meta.get("guideline", "")
                    if not guideline or guideline in seen:
                        continue
                    seen.add(guideline)
                    page = sr.get("PAGE_NUMBER_ESTIMATE") or meta.get("page", "")
                    citations.append({
                        "guideline": guideline,
                        "section": meta.get("section", ""),
                        "page": page,
                        "source_url": source_urls.get(guideline, ""),
                    })
        elif block.get("type") == "text":
            for ann in block.get("annotations", []) or []:
                text = ann.get("text", "")
                meta = parse_chunk_meta(text)
                guideline = meta.get("guideline", "")
                if guideline and guideline not in seen:
                    seen.add(guideline)
                    citations.append({
                        "guideline": guideline,
                        "section": meta.get("section", ""),
                        "page": meta.get("page", ""),
                        "source_url": source_urls.get(guideline, ""),
                    })
    return citations


def render_citations(citations):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for i, c in enumerate(citations, 1):
            page_txt = f", Page ~{c['page']}" if c.get("page") else ""
            section_txt = f" — {c['section']}" if c.get("section") and c["section"].upper() != "TEXAS CHILDREN'S HOSPITAL" else ""
            if c.get("source_url"):
                url = c["source_url"]
                if c.get("page"):
                    url += f"#page={c['page']}"
                st.markdown(f"{i}. [{c['guideline']}]({url}){section_txt}{page_txt}")
            else:
                st.markdown(f"{i}. {c['guideline']}{section_txt}{page_txt}")


# --- Agent Call ---
def run_agent(session, question, specialty, source_urls):
    context_prefix = ""
    if specialty and specialty != "All Specialties":
        context_prefix = f"[Category: {specialty}] "

    full_question = context_prefix + question

    request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_question}]}],
        "stream": False,
    }
    if st.session_state.thread_id:
        request["thread_id"] = st.session_state.thread_id
        request["parent_message_id"] = st.session_state.parent_message_id

    request_body = json.dumps(request)
    create_thread = "TRUE" if not st.session_state.thread_id else "FALSE"

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.DATA_AGENT_RUN(
            '{AGENT_NAME}',
            $${request_body}$$,
            {create_thread}
        ) AS response
    """
    result = session.sql(sql).collect()
    raw = result[0]["RESPONSE"] if result else None
    if not raw:
        return None, [], None

    parsed = json.loads(raw) if isinstance(raw, str) else raw

    # Extract thread/message IDs for multi-turn
    metadata = parsed.get("metadata", {})
    if metadata.get("thread_id"):
        st.session_state.thread_id = metadata["thread_id"]
    if metadata.get("assistant_message_id"):
        st.session_state.parent_message_id = metadata["assistant_message_id"]

    # Extract text response
    answer_text = ""
    for block in parsed.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            answer_text += block["text"] + "\n"

    citations = extract_citations(parsed, source_urls)
    return answer_text.strip(), citations, parsed


# --- Logging ---
def log_query(session, question, answer, specialty, citations, thread_id):
    cited_names = [c["guideline"] for c in citations] if citations else []
    session.sql(
        f"INSERT INTO {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"(SESSION_ID, THREAD_ID, SPECIALTY, QUESTION, RESPONSE_SUMMARY, CITED_GUIDELINES, LOG_TYPE) "
        f"SELECT ?, ?, ?, ?, ?, PARSE_JSON(?), 'query'",
        params=[
            st.session_state.session_id,
            thread_id or 0,
            specialty or "All",
            question,
            answer[:2000] if answer else "",
            json.dumps(cited_names),
        ]
    ).collect()


def log_feedback(session, question, feedback, feedback_text=""):
    session.sql(
        f"UPDATE {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"SET FEEDBACK = ?, FEEDBACK_TEXT = ? "
        f"WHERE SESSION_ID = ? AND QUESTION = ? AND LOG_TYPE = 'query' "
        f"ORDER BY CREATED_AT DESC LIMIT 1",
        params=[feedback, feedback_text, st.session_state.session_id, question]
    ).collect()


# --- UI Pages ---
def page_chat():
    session = get_active_session()
    source_urls = load_source_urls()

    # Version alerts banner
    alerts = load_version_alerts()
    if alerts:
        st.warning(f"⚠️ {len(alerts)} guideline(s) may have been updated in the past 7 days. Content may need re-indexing.")

    # Specialty filter
    col1, col2 = st.columns([3, 1])
    with col2:
        specialty = st.selectbox("Specialty", CATEGORIES, index=0, key="specialty_select")
    with col1:
        if st.session_state.thread_id:
            st.caption(f"Thread active — multi-turn conversation in progress")
            if st.button("New Conversation", type="secondary"):
                st.session_state.thread_id = None
                st.session_state.parent_message_id = 0
                st.session_state.messages = []
                st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg["role"] == "assistant" and msg.get("question"):
                col_a, col_b, col_c = st.columns([1, 1, 6])
                with col_a:
                    if st.button("👍", key=f"up_{msg['question'][:20]}"):
                        log_feedback(session, msg["question"], "positive")
                        st.toast("Thanks for the feedback!")
                with col_b:
                    if st.button("👎", key=f"down_{msg['question'][:20]}"):
                        log_feedback(session, msg["question"], "negative")
                        st.toast("Feedback recorded. Use the correction box below if you'd like to explain.")

    # Input
    if question := st.chat_input("Ask a clinical question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching EBOC guidelines..."):
                answer, citations, raw_response = run_agent(session, question, specialty, source_urls)

            if answer:
                st.markdown(answer)
                render_citations(citations)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "citations": citations, "question": question
                })
                log_query(session, question, answer, specialty, citations, st.session_state.thread_id)
            else:
                st.error("No response received from the agent. Please try again.")


def page_algorithms():
    st.subheader("Clinical Algorithms & Protocols")
    st.caption("Structured decision trees, dosing tables, and protocols extracted from EBOC guidelines.")

    algorithms = load_algorithms()
    if not algorithms:
        st.info("No algorithms extracted yet.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.selectbox("Category", ["All"] + sorted(set(r["CATEGORY"] for r in algorithms)))
    with col2:
        type_filter = st.selectbox("Type", ["All", "decision_tree", "dosing_table", "protocol_steps"])
    with col3:
        search_term = st.text_input("Search", placeholder="e.g., DKA, seizure, dosing")

    filtered = algorithms
    if cat_filter != "All":
        filtered = [a for a in filtered if a["CATEGORY"] == cat_filter]
    if type_filter != "All":
        filtered = [a for a in filtered if a["ALGORITHM_TYPE"] == type_filter]
    if search_term:
        term = search_term.lower()
        filtered = [a for a in filtered if term in (a["ALGORITHM_NAME"] or "").lower()
                    or term in (a["TEXT_SUMMARY"] or "").lower()
                    or term in (a["GUIDELINE_NAME"] or "").lower()]

    st.caption(f"Showing {len(filtered)} of {len(algorithms)} algorithms")

    for alg in filtered:
        type_icon = {"decision_tree": "🌳", "dosing_table": "💊", "protocol_steps": "📋"}.get(alg["ALGORITHM_TYPE"], "📄")
        with st.expander(f"{type_icon} {alg['ALGORITHM_NAME']} — {alg['GUIDELINE_NAME']}"):
            st.caption(f"Category: {alg['CATEGORY']} | Type: {alg['ALGORITHM_TYPE']}")
            if alg["TEXT_SUMMARY"]:
                st.markdown(f"**Summary:** {alg['TEXT_SUMMARY']}")

            content = alg["STRUCTURED_CONTENT"]
            if content:
                if isinstance(content, str):
                    content = json.loads(content)
                if isinstance(content, list):
                    for step in content:
                        step_num = step.get("step", "")
                        condition = step.get("condition", "")
                        action = step.get("action", "")
                        if condition and action:
                            st.markdown(f"**Step {step_num}:** IF {condition} → {action}")
                        elif action:
                            st.markdown(f"**Step {step_num}:** {action}")
                        elif condition:
                            st.markdown(f"**Step {step_num}:** {condition}")


def page_analytics():
    st.subheader("Usage Analytics")
    analytics = load_analytics()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", analytics["total"])
    with col2:
        positive = sum(r["CNT"] for r in analytics["feedback"] if r["FEEDBACK"] == "positive")
        st.metric("Positive Feedback", positive)
    with col3:
        negative = sum(r["CNT"] for r in analytics["feedback"] if r["FEEDBACK"] == "negative")
        st.metric("Negative Feedback", negative)

    if analytics["top_specialties"]:
        st.markdown("**Queries by Specialty**")
        for row in analytics["top_specialties"]:
            st.markdown(f"- {row['SPECIALTY']}: {row['CNT']} queries")

    # Recent queries
    session = get_active_session()
    recent = session.sql(
        f"SELECT CREATED_AT, SPECIALTY, QUESTION, LEFT(RESPONSE_SUMMARY, 100) AS ANSWER_PREVIEW, FEEDBACK "
        f"FROM {AGENT_DB}.{AGENT_SCHEMA}.QUERY_LOG "
        f"WHERE LOG_TYPE = 'query' ORDER BY CREATED_AT DESC LIMIT 20"
    ).collect()
    if recent:
        st.markdown("**Recent Queries**")
        st.dataframe(recent, use_container_width=True)


def page_guidelines():
    st.subheader("Guideline Index")
    st.caption("All 56 EBOC clinical practice guidelines with version tracking status.")

    session = get_active_session()
    guidelines = session.sql(
        f"SELECT GUIDELINE_NAME, CATEGORY, SOURCE_URL, LAST_CHECKED, VERSION_HASH "
        f"FROM {AGENT_DB}.{AGENT_SCHEMA}.CATEGORY_MAP ORDER BY CATEGORY, GUIDELINE_NAME"
    ).collect()

    cat_filter = st.selectbox("Filter by category", ["All"] + sorted(set(r["CATEGORY"] for r in guidelines)))
    filtered = guidelines if cat_filter == "All" else [g for g in guidelines if g["CATEGORY"] == cat_filter]

    for g in filtered:
        checked = g["LAST_CHECKED"].strftime("%Y-%m-%d %H:%M") if g["LAST_CHECKED"] else "Never"
        col1, col2 = st.columns([4, 1])
        with col1:
            if g["SOURCE_URL"]:
                st.markdown(f"**[{g['GUIDELINE_NAME']}]({g['SOURCE_URL']})** — {g['CATEGORY']}")
            else:
                st.markdown(f"**{g['GUIDELINE_NAME']}** — {g['CATEGORY']}")
        with col2:
            st.caption(f"Checked: {checked}")


# --- Main App ---
def main():
    st.set_page_config(
        page_title="EBOC Clinical Decision Support",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for mobile responsiveness
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .stColumns > div { min-width: 100% !important; }
        .stExpander { font-size: 0.9em; }
    }
    .stChatMessage { max-width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Navigation
    st.title("EBOC Clinical Decision Support")
    st.caption("Texas Children's Hospital Evidence-Based Outcomes Center")

    tabs = st.tabs(["💬 Clinical Assistant", "📋 Algorithms", "📊 Analytics", "📚 Guidelines"])

    with tabs[0]:
        page_chat()
    with tabs[1]:
        page_algorithms()
    with tabs[2]:
        page_analytics()
    with tabs[3]:
        page_guidelines()


if __name__ == "__main__":
    main()
