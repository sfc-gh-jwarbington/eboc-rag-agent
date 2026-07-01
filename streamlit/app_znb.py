import streamlit as st
from snowflake.snowpark.context import get_active_session
import json
import re

AGENTS = {
    "All Specialties": "EBOC_UNIVERSAL_AGENT",
    "Neurology": "EBOC_NEUROLOGY_AGENT",
    "Cardiology": "EBOC_CARDIOLOGY_AGENT",
    "Respiratory": "EBOC_RESPIRATORY_AGENT",
    "Infection": "EBOC_INFECTION_AGENT",
}

AGENT_DB = "EBOC_RAG"
AGENT_SCHEMA = "PUBLIC"


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_data(show_spinner=False)
def load_source_urls():
    """guideline_name -> source_url, from CATEGORY_MAP (for clickable citation links)."""
    session = get_active_session()
    rows = session.sql(
        f"SELECT guideline_name, source_url FROM {AGENT_DB}.{AGENT_SCHEMA}.CATEGORY_MAP"
    ).collect()
    return {r["GUIDELINE_NAME"]: r["SOURCE_URL"] for r in rows}


def parse_chunk_meta(text):
    """Pull guideline/category/section/subsection out of the chunk metadata prepend."""
    if not text:
        return {}
    meta = {}
    m = re.search(r"Guideline:\s*(.*?)\s*\|\s*Category:\s*([^\n]*)", text)
    if m:
        meta["guideline"] = m.group(1).strip()
        meta["category"] = m.group(2).strip()
    s = re.search(r"(?:^|\n)Section:\s*([^\n]*)", text)
    if s:
        meta["section"] = s.group(1).strip()
    sub = re.search(r"(?:^|\n)Subsection:\s*([^\n]*)", text)
    if sub:
        meta["subsection"] = sub.group(1).strip()
    return meta


def _collect_cited_chunks(parsed):
    """Prefer the text block's annotations (chunks the model actually cited);
    fall back to all retrieved search_results if there are no annotations."""
    annotated = []
    retrieved = []
    for block in parsed.get("content", []):
        btype = block.get("type")
        if btype == "text":
            for ann in block.get("annotations", []) or []:
                if ann.get("text"):
                    annotated.append(ann["text"])
        elif btype == "tool_result":
            for c in block.get("tool_result", {}).get("content", []) or []:
                j = c.get("json", {}) or {}
                for sr in j.get("search_results", []) or []:
                    if sr.get("text"):
                        retrieved.append(sr["text"])
    return annotated if annotated else retrieved


def build_citations(parsed, source_urls):
    """Return a deduped list of {guideline, section, source_url} for display."""
    citations = []
    seen = set()
    for text in _collect_cited_chunks(parsed):
        meta = parse_chunk_meta(text)
        guideline = meta.get("guideline", "")
        if not guideline:
            continue
        section = meta.get("section", "")
        key = (guideline, section)
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "guideline": guideline,
            "section": section,
            "source_url": source_urls.get(guideline, ""),
        })
    return citations


def render_citations(citations):
    if not citations:
        return
    with st.expander(f"📄 Sources ({len(citations)})", expanded=True):
        for i, c in enumerate(citations, 1):
            section = c.get("section", "")
            section_txt = f" — *{section}*" if section and section.upper() != "TEXAS CHILDREN'S HOSPITAL" else ""
            if c.get("source_url"):
                st.markdown(f"{i}. [{c['guideline']}]({c['source_url']}){section_txt}")
            else:
                st.markdown(f"{i}. {c['guideline']}{section_txt}")


def run_agent(session, agent_name, user_question, source_urls):
    fqn = f"{AGENT_DB}.{AGENT_SCHEMA}.{agent_name}"
    request_body = json.dumps({
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_question}]}
        ],
        "stream": False,
    })
    sql = f"""
        SELECT SNOWFLAKE.CORTEX.DATA_AGENT_RUN(
            '{fqn}',
            $${request_body}$$
        ) AS response
    """
    result = session.sql(sql).collect()[0]["RESPONSE"]
    parsed = json.loads(result)

    answer_parts = []
    for block in parsed.get("content", []):
        if block.get("type") == "text":
            answer_parts.append(block["text"])

    answer = "\n".join(answer_parts).strip()
    citations = build_citations(parsed, source_urls)
    return answer, citations


def main():
    st.set_page_config(
        page_title="EBOC Clinical Guidelines Assistant",
        page_icon="🏥",
        layout="wide",
    )

    st.title("EBOC Clinical Guidelines Assistant")
    st.caption("TCH — Evidence-Based Outcomes Center")

    with st.sidebar:
        st.header("Settings")
        specialty = st.selectbox("Specialty", list(AGENTS.keys()))
        st.divider()
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown(
            "**Disclaimer:** This tool is for reference only. "
            "Always verify recommendations with the full guideline document."
        )
        st.markdown("---")
        st.markdown(
            "**Sample Questions:**\n"
            "- What is the recommended workup for a first unprovoked seizure?\n"
            "- What are the diagnostic criteria for Kawasaki disease?\n"
            "- When should HFNC be initiated for bronchiolitis?\n"
            "- What antibiotics are recommended for community-acquired pneumonia?\n"
            "- What is the initial fluid resuscitation protocol for DKA?"
        )

    init_session_state()
    session = get_active_session()
    source_urls = load_source_urls()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    if question := st.chat_input("Ask a clinical question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Searching EBOC guidelines..."):
                agent_name = AGENTS[specialty]
                answer, citations = run_agent(session, agent_name, question, source_urls)
                st.markdown(answer)
                render_citations(citations)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
        })


if __name__ == "__main__":
    main()
