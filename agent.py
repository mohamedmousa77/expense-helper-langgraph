"""CLI chat agent built with LangGraph + Google Gemini.

Graph:  START -> llm -> (tools | END);  tools -> llm
State:  messages (with the add_messages reducer)
Memory: in-memory MemorySaver checkpointer, keyed by thread_id
"""

from __future__ import annotations

import argparse
import ast
import operator
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()  # reads GEMINI_API_KEY from .env

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST (no names, calls, or attributes)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(12.5 * 4) / 3 + 2**8'.

    Supports + - * / // % ** and parentheses.
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return str(result)
    except Exception as exc:  # noqa: BLE001 - report any failure back to the LLM
        return f"Error evaluating {expression!r}: {exc}"


_MOCK_RESULTS = {
    "weather": "Mock weather: 18°C, partly cloudy, light wind from the west.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor LLM "
    "applications as graphs, by the LangChain team.",
    "python": "Python is a high-level, general-purpose programming language. "
    "Latest stable release series: 3.14.",
}


@tool
def web_search(query: str) -> str:
    """Search the web for current information. (Mock implementation — no API key needed.)"""
    q = query.lower()
    hits = [text for key, text in _MOCK_RESULTS.items() if key in q]
    if hits:
        return "\n".join(f"- {h}" for h in hits)
    return f"- (mock) No live results for {query!r}. This is a stubbed search tool."


@tool
def eur_to_usd(amount: float) -> str:
    """Convert an amount in EUR to USD using a fixed reference rate (1 EUR = 1.08 USD)."""
    rate = 1.08
    usd = amount * rate
    return f"{amount:.2f} EUR = {usd:.2f} USD (rate: {rate})"


TOOLS = [calculator, web_search, eur_to_usd]

# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the calculator tool for any arithmetic "
    "and the web_search tool for facts you don't know or that may be current."
)


def build_graph(model: str = "gemini-3.5-flash-lite"):
    llm = ChatGoogleGenerativeAI(model=model, temperature=0).bind_tools(TOOLS)

    def llm_node(state: State) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    def route_after_llm(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(State)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", END: END})
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

HELP = "Commands: /new (new thread), /thread <id> (switch thread), /quit"


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph CLI chat agent")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--thread", default=None, help="thread_id to start with")
    args = parser.parse_args()

    graph = build_graph(args.model)

    print("=== Graph (Mermaid) ===")
    print(graph.get_graph().draw_mermaid())
    print("=======================\n")

    thread_id = args.thread or uuid.uuid4().hex[:8]
    print(f"thread_id = {thread_id}   ({HELP})\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        if user == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"[new thread_id = {thread_id}]")
            continue
        if user.startswith("/thread "):
            thread_id = user.split(maxsplit=1)[1]
            print(f"[switched to thread_id = {thread_id}]")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        try:
            for update in graph.stream(
                {"messages": [HumanMessage(user)]}, config, stream_mode="updates"
            ):
                for node, payload in update.items():
                    for msg in payload["messages"]:
                        if node == "tools":
                            print(f"  [tool {msg.name}] -> {msg.content}")
                        elif msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  [call {tc['name']}({tc['args']})]")
                        else:
                            print(f"bot> {msg.text}")
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            print(f"[error] {exc}")


if __name__ == "__main__":
    main()
