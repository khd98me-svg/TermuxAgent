import os
import tempfile
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


class ProviderRetryTests(unittest.TestCase):
    def setUp(self):
        self.old_bench = dict(agent.bench)
        self.old_streak = dict(agent.fail_streak)
        agent.bench.clear()
        agent.fail_streak.clear()

    def tearDown(self):
        agent.bench.clear()
        agent.bench.update(self.old_bench)
        agent.fail_streak.clear()
        agent.fail_streak.update(self.old_streak)

    def test_numeric_retry_after_is_parsed(self):
        self.assertEqual(agent.retry_after_seconds({"Retry-After": "45"}, now=1000), 45.0)

    def test_model_not_found_does_not_start_ten_minute_provider_bench(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "missing-model", 404)
        self.assertNotIn(("GROQ", "missing-model"), agent.bench)

    def test_payload_too_large_does_not_start_repeated_cooldown(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "model", 413)
        self.assertNotIn(("GROQ", "model"), agent.bench)

    def test_http_date_retry_after_is_parsed(self):
        from email.utils import formatdate
        self.assertEqual(agent.retry_after_seconds({"retry-after": formatdate(1120, usegmt=True)}, now=1000), 120.0)

    def test_429_uses_server_retry_after_instead_of_generic_bench(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "model", 429, 90)
        self.assertEqual(agent.bench[("GROQ", "model")], 1090)

    def test_429_without_header_uses_short_cooldown(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "model", 429)
        self.assertEqual(agent.bench[("GROQ", "model")], 1000 + agent.SHORT_COOLDOWN)

    def test_repeated_server_errors_use_bounded_backoff_not_ten_minutes(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "model", 503)
            agent._mark_failure("GROQ", "model", 503)
        self.assertLessEqual(agent.bench[("GROQ", "model")] - 1000, 120)

    def test_bad_request_does_not_bench_provider(self):
        with patch.object(agent.time, "time", return_value=1000):
            agent._mark_failure("GROQ", "model", 400)
        self.assertNotIn(("GROQ", "model"), agent.bench)

    def test_successful_request_clears_old_bench_and_streak(self):
        key = ("GROQ", agent.OPENAI["GROQ"]["model"])
        agent.bench[key] = 9999999999
        agent.fail_streak[key] = 2
        agent._mark_success(*key)
        self.assertNotIn(key, agent.bench)
        self.assertNotIn(key, agent.fail_streak)

    def test_provider_error_summary_includes_request_id_without_secret_headers(self):
        error = agent.ApiError(429, '{"error":{"message":"rate limited"}}', {
            "Retry-After": "30", "x-request-id": "req-123", "Authorization": "Bearer do-not-print"
        })
        summary = agent._provider_error_summary(error)
        self.assertIn("retry-after=30s", summary)
        self.assertIn("x-request-id=req-123", summary)
        self.assertNotIn("do-not-print", summary)


class HttpErrorMetadataTests(unittest.TestCase):
    def test_http_error_preserves_response_headers(self):
        import io
        error = agent.urllib.error.HTTPError(
            "https://example.invalid", 429, "Too Many Requests",
            {"Retry-After": "17", "x-request-id": "req-456"},
            io.BytesIO(b'{"error":"limited"}')
        )
        with patch.object(agent.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(agent.ApiError) as caught:
                agent._http("https://example.invalid", {}, {}, 1)
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.headers.get("retry-after"), "17")
        self.assertEqual(caught.exception.headers.get("x-request-id"), "req-456")


class MakeDirectoryApprovalTests(unittest.TestCase):
    def test_directory_creation_is_refused_without_approval(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "new-folder")
            with patch.object(agent, "ROOT", root):
                with patch.object(agent, "ask", return_value=False) as ask:
                    with patch.object(agent, "schedule_repo_map_refresh"):
                        with patch.object(agent, "journal", []):
                            result = agent.t_make_dir("new-folder")
            ask.assert_called_once()
            self.assertIn("denied", result.lower())
            self.assertFalse(os.path.exists(target))

    def test_directory_creation_proceeds_after_approval(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "new-folder")
            with patch.object(agent, "ROOT", root):
                with patch.object(agent, "ask", return_value=True) as ask:
                    with patch.object(agent, "schedule_repo_map_refresh"):
                        with patch.object(agent, "journal", []):
                            result = agent.t_make_dir("new-folder")
            ask.assert_called_once()
            self.assertIn("Created", result)
            self.assertTrue(os.path.isdir(target))


class ToolCallSafetyTests(unittest.TestCase):
    def test_malformed_json_arguments_are_marked_and_blocked(self):
        result = agent.from_openai({
            "choices": [{"message": {"tool_calls": [{
                "function": {"name": "list_dir", "arguments": "{not-json"}
            }]}}]
        })
        call = result["candidates"][0]["content"]["parts"][0]["functionCall"]
        self.assertTrue(call["invalidArguments"])
        with patch.dict(agent.TOOLS, {"list_dir": lambda **kwargs: self.fail("tool must not execute")}):
            with patch.object(agent, "out"), patch.object(agent, "log_line"):
                message = agent.execute(call)
        self.assertIn("execution was blocked", message)

    def test_non_object_json_arguments_are_marked_invalid(self):
        result = agent.from_openai({
            "choices": [{"message": {"tool_calls": [{
                "function": {"name": "list_dir", "arguments": '["unexpected", "array"]'}
            }]}}]
        })
        call = result["candidates"][0]["content"]["parts"][0]["functionCall"]
        self.assertTrue(call["invalidArguments"])
        self.assertEqual(call["args"], {})

    def test_stop_prevents_later_tools_in_same_batch(self):
        agent.history.clear()
        agent.STOP.clear()
        response = {"candidates": [{
            "content": {"role": "model", "parts": [
                {"functionCall": {"name": "first_tool", "args": {}}},
                {"functionCall": {"name": "second_tool", "args": {}}},
            ]}
        }]}
        calls = []
        def fake_execute(call):
            calls.append(call["name"])
            agent.STOP.set()
            return "first completed"
        try:
            with patch.object(agent, "call_llm", return_value=response), \
                 patch.object(agent, "execute", side_effect=fake_execute), \
                 patch.object(agent, "save_session"):
                agent.run_turn("stop during tool batch")
            self.assertEqual(calls, ["first_tool"])
            self.assertEqual(agent.history[0]["parts"][0]["text"], "stop during tool batch")
            self.assertIn("Some actions may already have completed", agent.history[-1]["parts"][0]["text"])
        finally:
            agent.STOP.clear()
            agent.history.clear()

    def test_interrupt_emits_stopped_event_and_preserves_auditable_history(self):
        agent.history.clear()
        agent.STOP.clear()
        response = {"candidates": [{
            "content": {"role": "model", "parts": [
                {"functionCall": {"name": "one_tool", "args": {}}},
                {"functionCall": {"name": "two_tool", "args": {}}},
            ]}
        }]}
        events = []
        def fake_execute(call):
            agent.STOP.set()
            return "completed"
        try:
            with patch.object(agent, "call_llm", return_value=response), \
                 patch.object(agent, "execute", side_effect=fake_execute), \
                 patch.object(agent, "out", side_effect=lambda kind, **kw: events.append((kind, kw))), \
                 patch.object(agent, "save_session"):
                agent.run_turn("retain interrupted request")
            self.assertIn(("stopped", unittest.mock.ANY), events)
            self.assertEqual(agent.history[0]["parts"][0]["text"], "retain interrupted request")
            self.assertEqual(agent.history[-1]["role"], "model")
        finally:
            agent.STOP.clear()
            agent.history.clear()


if __name__ == "__main__":
    unittest.main()
