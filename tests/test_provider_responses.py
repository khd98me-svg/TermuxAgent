import unittest
from unittest.mock import patch

import termux_agent as agent


class FromOpenAITests(unittest.TestCase):
    def test_text_content_is_preserved(self):
        result = agent.from_openai({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]
        })
        self.assertEqual(result["candidates"][0]["content"]["parts"], [{"text": "hello"}])

    def test_text_content_blocks_are_joined(self):
        result = agent.from_openai({
            "choices": [{"message": {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]}}]
        })
        self.assertEqual(result["candidates"][0]["content"]["parts"], [{"text": "hello world"}])

    def test_tool_calls_are_preserved(self):
        result = agent.from_openai({
            "choices": [{"message": {"tool_calls": [{
                "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}
            }]}}]
        })
        self.assertEqual(result["candidates"][0]["content"]["parts"], [
            {"functionCall": {"name": "list_dir", "args": {"path": "."}}}
        ])

    def test_refusal_is_visible_as_text(self):
        result = agent.from_openai({
            "choices": [{"message": {"refusal": "I cannot help with that request."}}]
        })
        self.assertEqual(result["candidates"][0]["content"]["parts"], [
            {"text": "I cannot help with that request."}
        ])

    def test_reasoning_only_is_diagnosed_without_exposing_reasoning(self):
        result = agent.from_openai({
            "choices": [{"message": {"content": None, "reasoning": "private reasoning"}, "finish_reason": "stop"}]
        })
        self.assertEqual(result["candidates"][0]["content"]["parts"], [])
        self.assertTrue(result["providerDiagnostics"]["reasoning_present"])
        self.assertNotIn("private reasoning", str(result))

    def test_missing_choices_is_reported(self):
        result = agent.from_openai({})
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["providerDiagnostics"]["choice_count"], 0)


class EmptyTurnTests(unittest.TestCase):
    def setUp(self):
        agent.history.clear()

    def tearDown(self):
        agent.history.clear()

    def test_empty_provider_response_is_visible_and_logged(self):
        response = {
            "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "stop"}],
            "providerDiagnostics": {"reasoning_present": True, "finish_reason": "stop"},
        }
        with patch.object(agent, "call_llm", return_value=response), \
             patch.object(agent, "out") as output, \
             patch.object(agent, "log_line") as log, \
             patch.object(agent, "save_session"):
            agent.run_turn("test")
        self.assertTrue(any(call.kwargs.get("text", "").startswith("The model returned reasoning")
                            for call in output.call_args_list))
        self.assertIn("reasoning but no user-visible answer", agent.history[-1]["parts"][0]["text"])
        log.assert_called_once()

    def test_missing_candidates_is_visible(self):
        with patch.object(agent, "call_llm", return_value={"candidates": []}), \
             patch.object(agent, "out") as output, \
             patch.object(agent, "log_line"), \
             patch.object(agent, "save_session"):
            agent.run_turn("test")
        self.assertTrue(any("no candidates" in call.kwargs.get("text", "") for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
