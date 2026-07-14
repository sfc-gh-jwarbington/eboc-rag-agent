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
    return [row.as_dict() for row in rows]


@st.cache_data(show_spinner=False)
def load_calculators():
    session = get_active_session()
    rows = session.sql(
        f"SELECT ID, GUIDELINE_NAME, CALCULATOR_NAME, CALCULATOR_TYPE, INPUTS, LOGIC "
        f"FROM {AGENT_DB}.{AGENT_SCHEMA}.CALCULATORS ORDER BY CALCULATOR_TYPE, CALCULATOR_NAME"
    ).collect()
    return [row.as_dict() for row in rows]


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


# --- Calculator Renderers ---
def render_calculator(calc, key_prefix=""):
    calc_type = calc["CALCULATOR_TYPE"]
    inputs = calc["INPUTS"] if isinstance(calc["INPUTS"], list) else json.loads(calc["INPUTS"]) if isinstance(calc["INPUTS"], str) else calc["INPUTS"]
    logic = calc["LOGIC"] if isinstance(calc["LOGIC"], dict) else json.loads(calc["LOGIC"]) if isinstance(calc["LOGIC"], str) else calc["LOGIC"]

    if calc_type == "checklist":
        render_checklist(calc["CALCULATOR_NAME"], inputs, logic, key_prefix)
    elif calc_type == "decision_tree":
        render_decision_tree(calc["CALCULATOR_NAME"], inputs, logic, key_prefix)
    elif calc_type == "dosing":
        render_dosing(calc["CALCULATOR_NAME"], inputs, logic, key_prefix)


def render_checklist(name, inputs, logic, key_prefix):
    threshold = logic.get("threshold", 1)
    checked_count = 0

    for inp in inputs:
        val = st.checkbox(inp["label"], key=f"{key_prefix}_{inp['name']}")
        if val:
            checked_count += 1

    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        color = "red" if checked_count >= threshold else "gray"
        st.markdown(f"### :{color}[{checked_count}/{len(inputs)}]")
    with col2:
        if checked_count >= threshold:
            st.success(logic.get("message_met", "Criteria met."))
        elif checked_count > 0 and logic.get("message_partial"):
            st.warning(logic["message_partial"])
        else:
            st.info(logic.get("message_not_met", "Criteria not met."))

    if logic.get("citation"):
        st.caption(logic["citation"])


def render_decision_tree(name, inputs, logic, key_prefix):
    responses = {}
    any_yes = False
    has_input = False

    for inp in inputs:
        if inp["type"] == "radio":
            val = st.radio(inp["label"], inp.get("options", ["Yes", "No"]),
                          key=f"{key_prefix}_{inp['name']}", horizontal=True)
            responses[inp["name"]] = val
            if val == "Yes":
                any_yes = True
            has_input = True
        elif inp["type"] == "number":
            val = st.number_input(inp["label"], min_value=0.0,
                                 key=f"{key_prefix}_{inp['name']}",
                                 help=inp.get("unit", ""))
            responses[inp["name"]] = val
            if val > 0:
                has_input = True

    st.divider()

    if not has_input:
        st.info("Fill in the fields above to get a recommendation.")
        return

    # Evaluate logic rules
    if logic.get("rules"):
        recommendation = evaluate_rules(responses, logic["rules"], logic)
        if recommendation:
            if any(kw in recommendation.lower() for kw in ["cancel", "admit", "immediate", "emergent", "stat", "toxic"]):
                st.error(recommendation)
            elif any(kw in recommendation.lower() for kw in ["consider", "consult", "moderate"]):
                st.warning(recommendation)
            else:
                st.success(recommendation)
        else:
            st.info("No matching recommendation for the current inputs.")
    elif logic.get("any_yes_recommendation"):
        if any_yes:
            st.error(logic["any_yes_recommendation"])
        else:
            st.success(logic["all_no_recommendation"])

    if logic.get("citation"):
        st.caption(logic["citation"])


def evaluate_rules(responses, rules, logic):
    for rule in rules:
        condition = rule.get("condition", "")
        if not condition:
            continue
        try:
            if matches_condition(responses, condition):
                return rule.get("recommendation", "")
        except (ValueError, TypeError, IndexError):
            continue
    # Fallback
    return logic.get("below_threshold", logic.get("all_no_recommendation", ""))


def matches_condition(responses, condition):
    parts = [p.strip() for p in condition.split(" AND ")]
    for part in parts:
        if " in " in part:
            field = part.split(" in ")[0].strip()
            values = [v.strip().strip("()") for v in part.split(" in ")[1].strip("()").split(",")]
            if field in responses and responses[field] in values:
                continue
            else:
                return False
        elif ">=" in part:
            field = part.split(">=")[0].strip()
            threshold = float(part.split(">=")[1].strip())
            if field in responses:
                try:
                    if float(responses[field]) >= threshold:
                        continue
                except (ValueError, TypeError):
                    return False
            return False
        elif "<=" in part:
            field = part.split("<=")[0].strip()
            threshold = float(part.split("<=")[1].strip())
            if field in responses:
                try:
                    if float(responses[field]) <= threshold:
                        continue
                except (ValueError, TypeError):
                    return False
            return False
        elif ">" in part and "=" not in part:
            field = part.split(">")[0].strip()
            threshold = float(part.split(">")[1].strip())
            if field in responses:
                try:
                    if float(responses[field]) > threshold:
                        continue
                except (ValueError, TypeError):
                    return False
            return False
        elif "<" in part and "=" not in part:
            field = part.split("<")[0].strip()
            threshold = float(part.split("<")[1].strip())
            if field in responses:
                try:
                    if float(responses[field]) < threshold:
                        continue
                except (ValueError, TypeError):
                    return False
            return False
        elif "=" in part:
            field, value = part.split("=", 1)
            field = field.strip()
            value = value.strip()
            if field in responses:
                if str(responses[field]) != value:
                    return False
            else:
                return False
        else:
            return False
    return True


def render_dosing(name, inputs, logic, key_prefix):
    values = {}
    for inp in inputs:
        if inp["type"] == "number":
            val = st.number_input(inp["label"], min_value=0.0, step=0.1,
                                 key=f"{key_prefix}_{inp['name']}",
                                 help=inp.get("unit", ""))
            values[inp["name"]] = val
        elif inp["type"] == "radio":
            val = st.radio(inp["label"], inp.get("options", []),
                          key=f"{key_prefix}_{inp['name']}", horizontal=True)
            values[inp["name"]] = val

    st.divider()

    if logic.get("ranges"):
        field = logic.get("field", "weight")
        val = values.get(field, 0)
        if val > 0:
            result = None
            for r in logic["ranges"]:
                if r["min"] <= val <= r["max"]:
                    result = r
                    break
            if result:
                st.success(f"**Dose:** {result['dose']}")
                if result.get("bottles"):
                    st.info(f"**Bottles:** {result['bottles']}")
            else:
                st.warning("Value outside defined ranges.")
        else:
            st.info("Enter patient weight to calculate dose.")
    elif logic.get("formula"):
        weight = values.get("weight", 0)
        dehydration = values.get("dehydration", "")
        if weight > 0 and dehydration:
            rates = logic.get("rates", {})
            key = dehydration.split("(")[1].rstrip("%)") if "(" in dehydration else "5"
            pct_map = {"5": "mild", "7": "moderate", "10": "severe"}
            rate_key = pct_map.get(key.strip("%"), "mild")
            rate = rates.get(rate_key, {})
            pct = rate.get("pct", 0.05)
            hours = rate.get("hours", 48)
            deficit_ml = weight * pct * 1000
            hourly_rate = deficit_ml / hours

            st.success(f"**Fluid deficit:** {deficit_ml:.0f} mL")
            st.info(f"**Replace over:** {hours} hours ({hourly_rate:.1f} mL/hr after bolus)")
            if logic.get("initial_bolus"):
                bolus = weight * 20
                st.warning(f"**Initial bolus (if unstable):** {bolus:.0f} mL (20 mL/kg NS over 1 hour)")
            if logic.get("replacement"):
                st.caption(logic["replacement"])
        else:
            st.info("Enter weight and dehydration severity to calculate.")

    if logic.get("citation"):
        st.caption(logic["citation"])


def find_calculators_for_guidelines(cited_guidelines, all_calculators):
    matches = []
    cited_lower = {g.lower() for g in cited_guidelines}
    for calc in all_calculators:
        if calc["GUIDELINE_NAME"].lower() in cited_lower:
            matches.append(calc)
    return matches


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

    metadata = parsed.get("metadata", {})
    if metadata.get("thread_id"):
        st.session_state.thread_id = metadata["thread_id"]
    if metadata.get("assistant_message_id"):
        st.session_state.parent_message_id = metadata["assistant_message_id"]

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
    all_calculators = load_calculators()

    alerts = load_version_alerts()
    if alerts:
        st.warning(f"{len(alerts)} guideline(s) may have been updated in the past 7 days. Content may need re-indexing.")

    col1, col2 = st.columns([3, 1])
    with col2:
        specialty = st.selectbox("Specialty", CATEGORIES, index=0, key="specialty_select")
    with col1:
        if st.session_state.thread_id:
            st.caption("Thread active — multi-turn conversation in progress")
            if st.button("New Conversation"):
                st.session_state.thread_id = None
                st.session_state.parent_message_id = 0
                st.session_state.messages = []
                st.rerun()

    # Chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])
            # Feedback buttons
            if msg["role"] == "assistant" and msg.get("question"):
                col_a, col_b, col_c = st.columns([1, 1, 6])
                with col_a:
                    if st.button("👍", key=f"up_{idx}"):
                        log_feedback(session, msg["question"], "positive")
                        st.toast("Thanks for the feedback!")
                with col_b:
                    if st.button("👎", key=f"down_{idx}"):
                        log_feedback(session, msg["question"], "negative")
                        st.toast("Feedback recorded.")
            # Calculators for this response
            if msg["role"] == "assistant" and msg.get("citations"):
                cited_guidelines = [c["guideline"] for c in msg["citations"]]
                matching_calcs = find_calculators_for_guidelines(cited_guidelines, all_calculators)
                for calc in matching_calcs:
                    type_icon = {"checklist": "✅", "decision_tree": "🔀", "dosing": "💊"}.get(calc["CALCULATOR_TYPE"], "🧮")
                    with st.expander(f"{type_icon} Calculator: {calc['CALCULATOR_NAME']}", expanded=False):
                        render_calculator(calc, key_prefix=f"chat_{idx}_{calc['ID']}")

    # Chat input
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

                # Auto-show relevant calculators
                cited_guidelines = [c["guideline"] for c in citations]
                matching_calcs = find_calculators_for_guidelines(cited_guidelines, all_calculators)
                for calc in matching_calcs:
                    type_icon = {"checklist": "✅", "decision_tree": "🔀", "dosing": "💊"}.get(calc["CALCULATOR_TYPE"], "🧮")
                    with st.expander(f"{type_icon} Calculator: {calc['CALCULATOR_NAME']}", expanded=True):
                        render_calculator(calc, key_prefix=f"new_{calc['ID']}")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "citations": citations, "question": question
                })
                log_query(session, question, answer, specialty, citations, st.session_state.thread_id)
            else:
                st.error("No response received from the agent. Please try again.")


def page_calculators():
    st.subheader("Clinical Calculators")
    st.caption("Interactive decision support tools derived from EBOC guidelines. Select a calculator to guide clinical decisions.")

    all_calcs = load_calculators()
    if not all_calcs:
        st.info("No calculators available.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        type_filter = st.selectbox("Type", ["All", "checklist", "decision_tree", "dosing"],
                                   format_func=lambda x: {"All": "All Types", "checklist": "✅ Criteria Checklists",
                                                          "decision_tree": "🔀 Decision Trees", "dosing": "💊 Dosing Calculators"}.get(x, x))
    with col2:
        search = st.text_input("Search", placeholder="e.g., Kawasaki, DKA, fever")

    filtered = all_calcs
    if type_filter != "All":
        filtered = [c for c in filtered if c["CALCULATOR_TYPE"] == type_filter]
    if search:
        term = search.lower()
        filtered = [c for c in filtered if term in c["CALCULATOR_NAME"].lower() or term in c["GUIDELINE_NAME"].lower()]

    st.caption(f"{len(filtered)} calculator(s) available")

    for calc in filtered:
        type_icon = {"checklist": "✅", "decision_tree": "🔀", "dosing": "💊"}.get(calc["CALCULATOR_TYPE"], "🧮")
        with st.expander(f"{type_icon} {calc['CALCULATOR_NAME']} — {calc['GUIDELINE_NAME']}"):
            render_calculator(calc, key_prefix=f"standalone_{calc['ID']}")


def page_algorithms():
    st.subheader("Clinical Algorithms & Protocols")
    st.caption("Structured decision trees, dosing tables, and protocols extracted from EBOC guidelines.")

    algorithms = load_algorithms()
    if not algorithms:
        st.info("No algorithms extracted yet.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.selectbox("Category", ["All"] + sorted(set(r["CATEGORY"] for r in algorithms)), key="alg_cat")
    with col2:
        type_filter = st.selectbox("Type", ["All", "decision_tree", "dosing_table", "protocol_steps"], key="alg_type")
    with col3:
        search_term = st.text_input("Search", placeholder="e.g., DKA, seizure, dosing", key="alg_search")

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
            if alg.get("TEXT_SUMMARY"):
                st.markdown(f"**Summary:** {alg['TEXT_SUMMARY']}")
            content = alg.get("STRUCTURED_CONTENT")
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
    guidelines = [row.as_dict() for row in session.sql(
        f"SELECT GUIDELINE_NAME, CATEGORY, SOURCE_URL, LAST_CHECKED, VERSION_HASH "
        f"FROM {AGENT_DB}.{AGENT_SCHEMA}.CATEGORY_MAP ORDER BY CATEGORY, GUIDELINE_NAME"
    ).collect()]

    cat_filter = st.selectbox("Filter by category", ["All"] + sorted(set(r["CATEGORY"] for r in guidelines)), key="guide_cat")
    filtered = guidelines if cat_filter == "All" else [g for g in guidelines if g["CATEGORY"] == cat_filter]

    for g in filtered:
        checked = g["LAST_CHECKED"].strftime("%Y-%m-%d %H:%M") if g.get("LAST_CHECKED") else "Never"
        col1, col2 = st.columns([4, 1])
        with col1:
            if g.get("SOURCE_URL"):
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

    st.title("EBOC Clinical Decision Support")
    st.caption("Texas Children's Hospital Evidence-Based Outcomes Center")

    tabs = st.tabs(["💬 Clinical Assistant", "🧮 Calculators", "📋 Algorithms", "📊 Analytics", "📚 Guidelines"])

    with tabs[0]:
        page_chat()
    with tabs[1]:
        page_calculators()
    with tabs[2]:
        page_algorithms()
    with tabs[3]:
        page_analytics()
    with tabs[4]:
        page_guidelines()


if __name__ == "__main__":
    main()
