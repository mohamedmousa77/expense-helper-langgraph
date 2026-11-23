"""Streamlit UI for the LangGraph + Google Gemini agent."""

from __future__ import annotations

import uuid

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from agent import build_graph

st.set_page_config(page_title="Expense Helper AI", page_icon="💬", layout="centered")

# --------------------------------------------------------------------------- #
# Custom Light-Mode Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.25rem 0;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: #475569;
        margin: 0 0 0.85rem 0;
    }
    .tool-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .tool-badge {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session State
# --------------------------------------------------------------------------- #
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]

if "history" not in st.session_state:
    st.session_state.history = {}


def _format_tool_args(args: dict) -> str:
    """Format tool arguments cleanly without raw JSON clutter."""
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _run_agent(prompt: str, tid: str) -> None:
    """Send a prompt to the LangGraph agent and append the turn to history."""
    messages = st.session_state.history.setdefault(tid, [])
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": tid}}
    tool_logs: list[dict[str, str]] = []
    pending_args: dict[str, str] = {}
    final_reply = ""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for update in st.session_state.graph.stream(
                {"messages": [HumanMessage(prompt)]}, config, stream_mode="updates"
            ):
                for node, payload in update.items():
                    for m in payload["messages"]:
                        if node == "tools":
                            args_str = pending_args.pop(m.name, "")
                            tool_logs.append(
                                {
                                    "name": m.name,
                                    "args": args_str,
                                    "output": str(m.content),
                                }
                            )
                        elif getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                pending_args[tc["name"]] = _format_tool_args(
                                    tc.get("args", {})
                                )
                        else:
                            final_reply = getattr(m, "text", None) or str(m.content)

        if tool_logs:
            label = "Used tool: " + ", ".join(t["name"] for t in tool_logs)
            with st.expander(label, expanded=False):
                for t in tool_logs:
                    detail = f"**{t['name']}**({t['args']}) → {t['output']}"
                    st.markdown(detail)
        st.markdown(final_reply)

    messages.append({"role": "assistant", "content": final_reply, "tools": tool_logs})


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
    st.subheader("Agent Graph")
    mermaid_diagram = (
        "graph TD; "
        "START((Start)) --> llm[LLM: Gemini]; "
        "llm -->|tool call| tools[Tools]; "
        "tools --> llm; "
        "llm -->|done| END((End));"
    )
    components.html(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:12px;display:flex;justify-content:center;">
          <div class="mermaid">{mermaid_diagram}</div>
        </div>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js';
          mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        """,
        height=320,
        scrolling=False,
    )

# --------------------------------------------------------------------------- #
# Header Card
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">Expense Helper AI</div>
      <div class="hero-sub">Stateful LangGraph + Gemini assistant with persistent thread memory.</div>
      <div class="tool-badges">
        <span class="tool-badge">🧮 Calculator</span>
        <span class="tool-badge">🌐 Web Search</span>
        <span class="tool-badge">💱 EUR → USD</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Chat History & Quick-Action Starter Chips
# --------------------------------------------------------------------------- #
tid = st.session_state.thread_id
messages = st.session_state.history.setdefault(tid, [])

for msg in messages:
    with st.chat_message(msg["role"]):
        if msg.get("tools"):
            label = "Used tool: " + ", ".join(t["name"] for t in msg["tools"])
            with st.expander(label, expanded=False):
                for t in msg["tools"]:
                    st.markdown(f"**{t['name']}**({t['args']}) → {t['output']}")
        st.markdown(msg["content"])

if not messages:
    starters = [
        "15% of 2,340 + 22% VAT",
        "Convert 250 EUR to USD",
        "Weather in Rome",
    ]
    cols = st.columns(len(starters))
    for col, starter in zip(cols, starters):
        if col.button(starter, use_container_width=True):
            _run_agent(starter, tid)
            st.rerun()

# --------------------------------------------------------------------------- #
# Chat Input
# --------------------------------------------------------------------------- #
if prompt := st.chat_input("Ask me anything..."):
    _run_agent(prompt, tid)
