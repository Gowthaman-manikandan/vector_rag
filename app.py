from __future__ import annotations

from pathlib import Path

import streamlit as st

import rag
from rag import CHROMA_DIR, DATA_DIR, answer, get_collection, ingest, retrieve


PROJECT_DIR = Path(__file__).resolve().parent


def display_path(path: Path) -> str:
    absolute_path = path.resolve()
    try:
        return str(absolute_path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(absolute_path)


st.set_page_config(page_title="Vector RAG Demo", page_icon="search", layout="wide")

st.title("Vector RAG Demo")
st.caption("Ask questions over local PDFs, Markdown, and text files in the data folder.")

with st.sidebar:
    st.header("Index")
    st.write(f"Data folder: `{DATA_DIR}`")
    st.write(f"Vector store: `{CHROMA_DIR}`")
    default_provider = rag.LLM_PROVIDER if rag.LLM_PROVIDER in rag.LLM_PROVIDERS else "ollama"
    provider = st.selectbox(
        "LLM provider",
        options=rag.LLM_PROVIDERS,
        index=rag.LLM_PROVIDERS.index(default_provider),
        format_func=str.title,
    )
    if provider == "ollama":
        st.write(f"Ollama URL: `{rag.OLLAMA_BASE_URL}`")
        model_name = st.text_input("Ollama model", value=rag.OLLAMA_MODEL)
    else:
        st.write(f"DeepSeek URL: `{rag.DEEPSEEK_BASE_URL}`")
        model_name = st.text_input("DeepSeek model", value=rag.DEEPSEEK_MODEL)

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        DATA_DIR.mkdir(exist_ok=True)
        for uploaded_file in uploaded_files:
            destination = DATA_DIR / uploaded_file.name
            destination.write_bytes(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index", use_container_width=True):
            with st.spinner("Embedding documents..."):
                count = ingest()
            st.success(f"Indexed {count} chunks.")
    with col2:
        if st.button("Rebuild", use_container_width=True):
            with st.spinner("Rebuilding vector store..."):
                count = ingest(reset=True)
            st.success(f"Rebuilt {count} chunks.")

    try:
        collection_count = get_collection().count()
    except Exception:
        collection_count = 0
    st.metric("Chunks", collection_count)

    st.subheader("Files")
    files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file())
    if files:
        for path in files:
            st.write(f"- {display_path(path)}")
    else:
        st.write("No documents yet.")


question = st.text_input("Question", placeholder="What is this document about?")
n_results = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
use_agent = st.checkbox("Enable Enterprise Assistant (Agentic Workflow)", value=True, help="Allows the AI to automatically route requests to Email, Jira, and Vector Search.")

if st.button("Ask", type="primary") and question:
    if use_agent:
        from agent import execute_agent
        with st.spinner("Agent is planning and executing steps..."):
            steps = execute_agent(question)
            
        st.subheader("Agent Workflow")
        for step in steps:
            tool_name = step.get("tool")
            output = step.get("output", "")
            details = step.get("details", {})
            
            if tool_name == "vector_search":
                st.markdown("**🔍 Vector Search**")
                st.info(output)
            # elif tool_name == "email_tool":
            #     st.markdown("**📧 Email Tool**")
            #     st.json(details)
            #     st.success(output)
            # elif tool_name == "jira_tool":
            #     st.markdown("**🎫 Jira Tool**")
            #     st.json(details)
            #     st.success(output)
            elif tool_name == "notion_tool":
                st.markdown("**📓 Notion Tool**")
                st.json(details)
                st.success(output)
            elif tool_name == "notion_ticket_tool":
                st.markdown("**📋 Notion Ticket Created**")
                if details:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Assignee", details.get("assignee_name", "Unassigned"))
                    with col2:
                        st.metric("Status", details.get("status", "To Do"))
                    with col3:
                        st.metric("Priority", details.get("priority", "Medium"))
                    st.markdown(f"**Title:** {details.get('title', '')}")
                    st.markdown(f"**Description:** {details.get('description', '')}")
                st.success(output)
            elif tool_name == "FINAL_ANSWER":
                st.markdown("**✨ Final Summary**")
                st.write(output)
    else:
        with st.spinner("Searching and composing answer..."):
            result = answer(
                question,
                n_results=n_results,
                provider=provider,
                model=model_name,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for index, source in enumerate(result["sources"], start=1):
            metadata = source["metadata"]
            label = f"{index}. {metadata.get('source')} page {metadata.get('page')}"
            with st.expander(label):
                st.write(source["text"])

elif question and not use_agent:
    with st.expander("Preview retrieved chunks"):
        for index, source in enumerate(retrieve(question, n_results=n_results), start=1):
            metadata = source["metadata"]
            st.markdown(f"**{index}. {metadata.get('source')} page {metadata.get('page')}**")
            st.write(source["text"])
