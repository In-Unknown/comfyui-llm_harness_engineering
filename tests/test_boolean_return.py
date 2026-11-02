"""Conversational retries through the real executor; no remote model requests."""

from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness_test_support import pack, reply, generation_cache
from test_image_inputs import DEFAULTS, ImageSource, decoded_images
from string_node import LiuxinyuParseJSONRecords

import execution
import nodes
import server
from comfy_api.latest import io
from comfy.model_management import InterruptProcessingException, interrupt_current_processing


class ValidateJSON(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="RetryTestValidate", inputs=[io.String.Input("text"), io.Int.Input("upper")],
                         outputs=[io.Boolean.Output(), io.String.Output()])

    @classmethod
    def execute(cls, text, upper):
        cls.seen.append(text)
        try:
            value = json.loads(text)["value"]
            passed = type(value) is int and 0 <= value <= upper
            error = "" if passed else f"value 必须为 0 到 {upper} 的整数，实际为 {value}"
        except (ValueError, KeyError, TypeError):
            passed, error = False, "必须是含 value 字段的 JSON 对象"
        return io.NodeOutput(passed, error)


class ReplySink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="RetryTestSink", is_output_node=True,
                         inputs=[io.String.Input("text"), io.Int.Input("count")], outputs=[])

    @classmethod
    def execute(cls, text, count):
        cls.seen.append((text, count))
        return io.NodeOutput(ui={"text": [text], "count": [count]})


class UpperBound(io.ComfyNode):
    calls = 0

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="RetryTestBound", inputs=[], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        UpperBound.calls += 1
        return io.NodeOutput(10)


class Prompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="RetryTestPrompts", inputs=[], outputs=[io.String.Output(is_output_list=True)])

    @classmethod
    def execute(cls):
        return io.NodeOutput(["first", "second"])


class BooleanReturnTests(unittest.TestCase):
    def setUp(self):
        ValidateJSON.seen, ReplySink.seen, UpperBound.calls = [], [], 0
        interrupt_current_processing(False)
        self.addCleanup(interrupt_current_processing, False)
        self.responses = ['{"value": 5}']
        self.requests = []
        self.enterContext(patch("logging.error"))
        self.cfg = {"endpoint": "https://example.invalid/v1", "api_format": "Chat Completions", "models": ["test-model"]}
        self.enterContext(patch.object(reply, "_load_connections", return_value={"test": self.cfg}))
        self.enterContext(patch.object(generation_cache, "CACHE", generation_cache.GenerationCache()))
        self.enterContext(patch.object(server.PromptServer, "instance", SimpleNamespace(client_id=None, send_sync=lambda *args: None), create=True))
        self.post = self.enterContext(patch.object(reply, "_post_stream", side_effect=self.respond))
        self.enterContext(patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Unexpected network request")))
        classes = dict(pack.NODE_CLASS_MAPPINGS)
        for cls in (ValidateJSON, ReplySink, UpperBound, Prompts, ImageSource, LiuxinyuParseJSONRecords):
            classes[cls.define_schema().node_id] = cls
        self.enterContext(patch.dict(nodes.NODE_CLASS_MAPPINGS, classes))

    def respond(self, url, headers, body, timeout, fmt, node_id=None):
        self.requests.append((deepcopy(body), fmt))
        response = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        if isinstance(response, BaseException):
            raise response
        if fmt == "Chat Completions":
            return {"choices": [{"message": {"content": response}}]}
        if fmt == "Responses":
            return {"output_text": response}
        return {"content": [{"type": "text", "text": response}]}

    def graph(self, max_retries=3):
        return {
            "reply": {"class_type": "llm_remote_reply", "inputs": dict(DEFAULTS)},
            "bound": {"class_type": "RetryTestBound", "inputs": {}},
            "check": {"class_type": "RetryTestValidate", "inputs": {"text": ["reply", 0], "upper": ["bound", 0]}},
            "return": {"class_type": "LLMHarnessBooleanReturn", "inputs": {
                "schedule": ["reply", 1], "passed": ["check", 0], "error_message": ["check", 1], "max_retries": max_retries}},
            "sink": {"class_type": "RetryTestSink", "inputs": {"text": ["return", 0], "count": ["return", 1]}},
        }

    def execute(self, graph=None, outputs=None, executor=None):
        executor = executor or execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                        execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
        executor.execute(deepcopy(graph or self.graph()), f"retry-test-{len(self.requests)}", execute_outputs=outputs or ["return", "sink"])
        return executor

    def assert_failed(self, executor, message):
        self.assertFalse(executor.success)
        self.assertIn(message, str(executor.status_messages))
        self.assertEqual(ReplySink.seen, [])

    def test_pass_first_time_and_original_output_contract(self):
        result = self.execute()
        self.assertTrue(result.success, result.status_messages)
        self.assertEqual(ReplySink.seen, [('{"value": 5}', 0)])
        self.assertEqual(len(self.requests), 1)
        schema = reply.LLMRemoteReply.define_schema()
        self.assertEqual([out.id for out in schema.outputs], ["text", "schedule"])
        self.assertTrue(pack.NODE_CLASS_MAPPINGS["LLMHarnessReplyRetry"].define_schema().is_dev_only)

    def test_documented_json_validator_wiring_validates_and_retries(self):
        self.responses = ['[{"id":"A","count":"2"}]', '[{"id":"A","count":2}]']
        graph = self.graph(2)
        graph["check"] = {"class_type": "LiuxinyuParseJSONRecords", "inputs": {
            "text": ["reply", 0], "fields": '{"id":"nonempty_string","count":"integer"}', "allow_extra": False}}
        graph["return"]["inputs"].update(passed=["check", 1], error_message=["check", 2])
        graph["final_parse"] = deepcopy(graph["check"])
        graph["final_parse"]["inputs"]["text"] = ["return", 0]
        validation = asyncio.run(execution.validate_prompt("documented-example", deepcopy(graph), None))
        self.assertTrue(validation[0], validation)
        result = self.execute(graph, outputs=["return", "sink", "final_parse"])
        self.assertTrue(result.success, result.status_messages)
        self.assertEqual(ReplySink.seen, [(self.responses[-1], 1)])
        parsed = result.caches.outputs.get_local("final_parse").outputs
        self.assertEqual(parsed, [[[{'id': 'A', 'count': 2}]], [True], [""]])
        self.assertIn("count", self.requests[-1][0]["messages"][-1]["content"])
        graph["return"]["inputs"]["max_retries"] = -1
        validation = asyncio.run(execution.validate_prompt("negative-budget", graph, None))
        self.assertFalse(validation[0])

    def test_independent_reply_pairs_keep_their_own_context_and_budget(self):
        graph = self.graph(1)
        other = self.graph(0)
        for key, node in other.items():
            inputs = {name: ["second_" + value[0], value[1]] if isinstance(value, list) else value
                      for name, value in node["inputs"].items()}
            graph["second_" + key] = {"class_type": node["class_type"], "inputs": inputs}
        graph["second_reply"]["inputs"]["prompt"] = "second prompt"

        def respond(url, headers, body, timeout, fmt, node_id=None):
            self.requests.append((deepcopy(body), fmt))
            prompt = body["messages"][1]["content"][-1]["text"]
            text = '{"value": 8}' if prompt == "second prompt" else ('{"value": 6}' if len(body["messages"]) > 2 else "bad")
            return {"choices": [{"message": {"content": text}}]}

        self.post.side_effect = respond
        result = self.execute(graph, outputs=["return", "sink", "second_return", "second_sink"])
        self.assertTrue(result.success, result.status_messages)
        self.assertCountEqual(ReplySink.seen, [('{"value": 6}', 1), ('{"value": 8}', 0)])
        self.assertEqual(len(self.requests), 3)

    def test_all_formats_keep_system_original_prompt_images_and_full_error_history(self):
        for fmt in ("Chat Completions", "Responses", "Anthropic Messages"):
            with self.subTest(format=fmt):
                self.requests.clear()
                ValidateJSON.seen, ReplySink.seen, UpperBound.calls = [], [], 0
                self.cfg["api_format"] = fmt
                self.responses = ["not json", '{"value": 50}', '{"value": 5}']
                graph = self.graph(2)
                graph["image"] = {"class_type": "TestHarnessImageSource", "inputs": {}}
                graph["reply"]["inputs"]["images.图片1"] = ["image", 0]
                result = self.execute(graph)
                self.assertTrue(result.success, result.status_messages)
                self.assertEqual(ValidateJSON.seen, self.responses)
                self.assertEqual(ReplySink.seen, [('{"value": 5}', 2)])
                self.assertEqual(UpperBound.calls, 1)
                self.assertEqual(len(self.requests), 3)
                for index, (body, _) in enumerate(self.requests):
                    messages = body["input"] if fmt == "Responses" else body["messages"]
                    if fmt == "Chat Completions":
                        self.assertEqual(messages[0], {"role": "system", "content": "system"})
                        messages = messages[1:]
                    else:
                        self.assertEqual(body["instructions" if fmt == "Responses" else "system"], "system")
                    self.assertEqual(len(messages), 1 + 2 * index)
                    self.assertEqual(messages[0]["content"][-1]["text"], "describe")
                    for turn in range(index):
                        self.assertEqual(messages[1 + turn * 2]["role"], "assistant")
                        self.assertIn(self.responses[turn], str(messages[1 + turn * 2]["content"]))
                        expected = "JSON 对象" if turn == 0 else "实际为 50"
                        self.assertIn(expected, str(messages[2 + turn * 2]["content"]))
                    # Original images stay in the first user message, in every request.
                    image_body = deepcopy(body)
                    if fmt != "Responses":
                        image_body["messages"] = [messages[0]]
                    self.assertEqual(decoded_images((image_body, fmt)), [((3, 250), (255, 0, 0))])

    def test_zero_budget_and_exhaustion_stop_before_downstream(self):
        for limit in (0, 2):
            with self.subTest(limit=limit):
                self.requests.clear()
                generation_cache.CACHE = generation_cache.GenerationCache()
                self.responses = ["bad"]
                result = self.execute(self.graph(limit))
                self.assert_failed(result, f"已重试 {limit}/{limit} 次")
                self.assertEqual(len(self.requests), limit + 1)

    def test_return_is_executed_without_a_sink(self):
        self.responses = ["bad"]
        self.assert_failed(self.execute(self.graph(0), outputs=["return"]), "已重试 0/0 次")

    def test_reject_negative_noninteger_and_nonboolean_controls(self):
        for field, value, message in (("max_retries", -1, "非负整数"), ("max_retries", 1.5, "非负整数"),
                                      ("passed", "false", "BOOLEAN"), ("passed", 0, "BOOLEAN")):
            with self.subTest(field=field, value=value):
                graph = self.graph()
                graph["return"]["inputs"][field] = value
                self.assert_failed(self.execute(graph), message)

    def test_empty_error_has_actionable_fallback_and_constant_condition_uses_budget(self):
        graph = self.graph(1)
        graph["return"]["inputs"].update(passed=False, error_message="  ")
        result = self.execute(graph)
        self.assert_failed(result, "回复未通过校验")
        self.assertEqual(len(self.requests), 2)
        self.assertIn("回复未通过校验", str(self.requests[-1][0]))

    def test_initial_cache_is_reused_but_retry_is_fresh_on_every_queue(self):
        self.responses = ["bad", '{"value": 5}', '{"value": 6}']
        executor = self.execute()
        self.assertTrue(executor.success, executor.status_messages)
        executor = self.execute(executor=executor)
        self.assertTrue(executor.success, executor.status_messages)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(ReplySink.seen, [('{"value": 5}', 1), ('{"value": 6}', 1)])
        self.assertEqual(ValidateJSON.seen, ["bad", '{"value": 5}', "bad", '{"value": 6}'])

    def test_http_failure_and_interruption_are_not_validation_retries(self):
        for error in (RuntimeError("service unavailable"), InterruptProcessingException()):
            with self.subTest(error=type(error).__name__):
                self.requests.clear()
                generation_cache.CACHE = generation_cache.GenerationCache()
                self.responses = ["bad", error]
                result = self.execute()
                self.assertFalse(result.success)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(ReplySink.seen, [])

    def test_multiple_replies_are_not_silently_paired_with_one_boolean(self):
        graph = self.graph()
        graph["prompts"] = {"class_type": "RetryTestPrompts", "inputs": {}}
        graph["reply"]["inputs"]["prompt"] = ["prompts", 0]
        self.assert_failed(self.execute(graph), "多项执行列表")
        self.assertEqual(len(self.requests), 2)

    def test_schedule_does_not_include_credentials_or_input_tensors(self):
        self.cfg["api_key"] = "private-test-key"
        result = self.execute()
        self.assertTrue(result.success, result.status_messages)
        # Check the actual live outputs, not just the schema.
        cached = result.caches.outputs.get_local("reply")
        schedule = cached.outputs[1][0]
        self.assertIsInstance(schedule, reply.ReplySchedule)
        self.assertNotIn("private-test-key", repr(schedule))
        self.assertNotIn("api_key", schedule.request)
        self.assertEqual(schedule.images_b64, ())


if __name__ == "__main__":
    unittest.main()
