"""Streamlit UI for the LangGraph + Google Gemini CLI agent."""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from agent import build_graph

st.set_page_config(page_title="Expense Helper Agent", page_icon="💬", layout="centered")
st.title("Expense Helper Agent")

# Persist compiled graph (and its MemorySaver) across Streamlit reruns
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]

if "history" not in st.session_state:
    # Maps thread_id -> list of {"role": "user"|"assistant", "content": str, "tools": list[str]}
    st.session_state.history = {}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Session")
    new_tid = st.text_input("thread_id", value=st.session_state.thread_id)
    if new_tid and new_tid != st.session_state.thread_id:
        st.session_state.thread_id = new_tid
        st.rerun()

    if st.button("New Thread", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.rerun()

    st.divider()
    st.subheader("Graph (Mermaid)")
    mermaid_src = st.session_state.graph.get_graph().draw_mermaid()
    st.markdown(f"```mermaid\n{mermaid_src}\n```")

# --------------------------------------------------------------------------- #
# Chat History
# --------------------------------------------------------------------------- #
tid = st.session_state.thread_id
messages = st.session_state.history.setdefault(tid, [])

for msg in messages:
    with st.chat_message(msg["role"]):
        if msg.get("tools"):
            with st.expander(f"Tool calls ({len(msg['tools'])})", expanded=False):
                for entry in msg["tools"]:
                    st.markdown(entry)
        st.markdown(msg["content"])

# --------------------------------------------------------------------------- #
# Chat Input
# --------------------------------------------------------------------------- #
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": tid}}
    tool_logs: list[str] = []
    final_reply = ""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for update in st.session_state.graph.stream(
                {"messages": [HumanMessage(prompt)]}, config, stream_mode="updates"
            ):
                for node, payload in update.items():
                    for m in payload["messages"]:
                        if node == "tools":
                            tool_logs.append(f"**[{m.name}]** -> `{m.content}`")
                        elif getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                tool_logs.append(
                                    f"**call `{tc['name']}`**(`{tc['args']}`)"
                                )
                        else:
                            final_reply = getattr(m, "text", None) or str(m.content)

        if tool_logs:
            with st.expander(f"Tool calls ({len(tool_logs)})", expanded=False):
                for entry in tool_logs:
                    st.markdown(entry)
        st.markdown(final_reply)

    messages.append({"role": "assistant", "content": final_reply, "tools": tool_logs})
