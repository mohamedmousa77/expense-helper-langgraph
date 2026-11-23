"""Live verification tests for agent.py against Google Gemini."""

from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage

from agent import build_graph


def _run_turn(graph, prompt: str, thread_id: str) -> dict:
    """Invoke one user turn and collect nodes visited, tool calls, tool outputs, and final text."""
    config = {"configurable": {"thread_id": thread_id}}
    nodes: list[str] = []
    tool_calls: list[dict] = []
    tool_outputs: list[tuple[str, str]] = []
    final_text = ""

    for update in graph.stream(
        {"messages": [HumanMessage(prompt)]}, config, stream_mode="updates"
    ):
        for node, payload in update.items():
            nodes.append(node)
            for msg in payload["messages"]:
                if node == "tools":
                    tool_outputs.append((msg.name, str(msg.content)))
                elif getattr(msg, "tool_calls", None):
                    tool_calls.extend(msg.tool_calls)
                else:
                    final_text = getattr(msg, "text", None) or str(msg.content)

    return {
        "nodes": nodes,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "final_text": final_text,
    }


class TestAgentLive(unittest.TestCase):
    def test_1_tool_call_required(self) -> None:
        """Calculator tool runs and returns 428.22."""
        graph = build_graph()
        res = _run_turn(
            graph, "What's 15% of 2,340 plus 22% VAT?", thread_id="test-calc"
        )
        self.assertIn("tools", res["nodes"], f"Expected tools node to run: {res}")
        calc_names = [tc["name"] for tc in res["tool_calls"]]
        self.assertIn("calculator", calc_names)
        combined = " ".join(out for _, out in res["tool_outputs"]) + " " + res["final_text"]
        self.assertIn("428.22", combined)

    def test_2_no_tool_required(self) -> None:
        """Direct llm -> END path with zero tool calls."""
        graph = build_graph()
        res = _run_turn(
            graph, "Hello, what can you help me with?", thread_id="test-direct"
        )
        self.assertNotIn("tools", res["nodes"], f"Unexpected tools node: {res}")
        self.assertEqual(res["tool_calls"], [])
        self.assertTrue(res["final_text"].strip(), "Expected non-empty assistant reply")

    def test_3_multi_turn_memory(self) -> None:
        """Follow-up on thread_id='t1' retrieves Milan weather using prior context."""
        graph = build_graph()
        _run_turn(graph, "What's the weather in Rome?", thread_id="t1")
        res2 = _run_turn(graph, "And what about Milan?", thread_id="t1")

        self.assertIn("tools", res2["nodes"], f"Expected web_search tool call: {res2}")
        queries = " ".join(str(tc.get("args", {})) for tc in res2["tool_calls"]).lower()
        self.assertIn("milan", queries, f"Expected 'milan' in tool args: {res2['tool_calls']}")
        self.assertIn("milan", res2["final_text"].lower())

    def test_4_memory_isolation_fresh_saver(self) -> None:
        """A fresh MemorySaver() with thread_id='t1' does not recall Rome or Milan."""
        graph1 = build_graph()
        _run_turn(graph1, "What's the weather in Rome?", thread_id="t1")
        _run_turn(graph1, "And what about Milan?", thread_id="t1")

        # Fresh graph -> fresh MemorySaver() instance
        graph2 = build_graph()
        state = graph2.get_state({"configurable": {"thread_id": "t1"}})
        self.assertFalse(state.values.get("messages"), "Fresh MemorySaver should have no messages")

        res = _run_turn(graph2, "What cities did we just talk about?", thread_id="t1")
        reply_lower = res["final_text"].lower()
        self.assertNotIn("rome", reply_lower)
        self.assertNotIn("milan", reply_lower)

    def test_5_eur_to_usd(self) -> None:
        """EUR-to-USD tool converts 250 EUR to 270.00 USD."""
        graph = build_graph()
        res = _run_turn(graph, "Convert 250 EUR to USD", thread_id="test-eur-usd")
        self.assertIn("tools", res["nodes"], f"Expected tools node to run: {res}")
        self.assertIn("eur_to_usd", [tc["name"] for tc in res["tool_calls"]])
        combined = " ".join(out for _, out in res["tool_outputs"]) + " " + res["final_text"]
        self.assertIn("270", combined)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAgentLive)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n=== Pass/Fail Report ===")
    print(f"Ran: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
