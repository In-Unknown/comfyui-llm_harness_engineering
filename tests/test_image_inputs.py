"""Exercise ComfyUI list mapping and request bodies without contacting a model."""

import asyncio
import base64
from copy import deepcopy
import io
from pathlib import Path
import sys
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK.parent / "comfyui-liuxinyu-packs"))

from aiohttp import web
from PIL import Image
import torch
import execution
import folder_paths
import nodes
import server
from comfy_api.latest import IO
from comfy.model_management import InterruptProcessingException
from split_preview_nodes import LiuxinyuSplitLook

from harness_test_support import reply, generation_cache


DEFAULTS = dict(connection="test", model="test-model", prompt="describe", system_prompt="system",
                thinking="关闭", temperature=0.7, max_tokens=100, timeout=30)


def picture(color, width=3, height=2, batch=1):
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(batch, height, width, 3)


def decoded_images(request):
    body, fmt = request
    content = (body["input"][0]["content"] if fmt == "Responses"
               else body["messages"][-1]["content"])
    images = []
    for item in content:
        if item["type"] == "image_url":
            encoded = item["image_url"]["url"].split(",", 1)[1]
        elif item["type"] == "input_image":
            encoded = item["image_url"].split(",", 1)[1]
        elif item["type"] == "image":
            encoded = item["source"]["data"]
        else:
            continue
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            images.append((image.size, image.getpixel((0, 0))))
    return images


class ImageSource(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id="TestHarnessImageSource", inputs=[], outputs=[IO.Image.Output()])

    @classmethod
    def execute(cls):
        return IO.NodeOutput(picture([1, 0, 0], height=250))


class ReplySink(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id="TestHarnessReplySink", inputs=[IO.String.Input("text", force_input=True)],
                         outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, text):
        return IO.NodeOutput(ui={"text": [text]})


class ImageInputTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.enterContext(patch.object(generation_cache, "CACHE", generation_cache.GenerationCache()))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(folder_paths, "get_user_directory", return_value=self.temp))
        self.enterContext(patch.object(server.PromptServer, "instance", SimpleNamespace(client_id=None, send_sync=lambda *args: None), create=True))
        self.config = {"endpoint": "https://example.invalid", "api_format": "Chat Completions", "models": ["test-model"]}
        self.enterContext(patch.object(reply, "_load_connections", return_value={"test": self.config}))
        self.enterContext(patch.object(reply, "_post_stream", side_effect=self.respond))
        self.enterContext(patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network request")))

    def respond(self, url, headers, body, timeout, fmt, node_id=None):
        self.requests.append((body, fmt))
        text = f"reply-{len(self.requests)}"
        if fmt == "Chat Completions":
            return {"choices": [{"message": {"content": text}}]}
        if fmt == "Responses":
            return {"output_text": text}
        return {"content": [{"type": "text", "text": text}]}

    def run_node(self, node_id="reply", **inputs):
        live_inputs = {**DEFAULTS, **dict.fromkeys(inputs)}
        _, missing, v3_data = execution.get_input_data(live_inputs, reply.LLMRemoteReply, node_id)
        self.assertFalse(missing)
        mapped = {key: [value] for key, value in DEFAULTS.items()}
        mapped.update(inputs)
        outputs, _, _, _ = asyncio.run(execution.get_output_data(
            "image-input-test", node_id, reply.LLMRemoteReply, mapped, v3_data=v3_data))
        return outputs

    def test_merge_different_sizes_and_nested_batches_in_all_api_formats(self):
        for fmt in ("Chat Completions", "Responses", "Anthropic Messages"):
            with self.subTest(format=fmt):
                self.config["api_format"] = fmt
                self.requests.clear()
                result = self.run_node(image_list=[picture([1, 0, 0], 5, 7, batch=2), picture([0, 1, 0], 4, 2)])
                self.assertEqual(result, [["reply-1"]])
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(decoded_images(self.requests[0]),
                                 [((5, 7), (255, 0, 0)), ((5, 7), (255, 0, 0)), ((4, 2), (0, 255, 0))])

    def test_models_from_other_connections_validate_and_execute_with_images(self):
        connections = {
            "a-image": {**self.config, "models": ["gpt-image-2"]},
            "b-chat": {**self.config, "models": ["gpt-5.6-sol", "gpt-6-sol", "gpt-6-astra"]},
            "c-legacy": {**self.config, "models": [], "model": "legacy-model"},
        }
        classes = {"llm_remote_reply": reply.LLMRemoteReply, "TestHarnessImageSource": ImageSource,
                   "TestHarnessReplySink": ReplySink}
        with patch.object(reply, "_load_connections", return_value=connections), patch.dict(nodes.NODE_CLASS_MAPPINGS, classes):
            for connection, models in (("b-chat", connections["b-chat"]["models"]), ("c-legacy", ["legacy-model"])):
                for model in models:
                    with self.subTest(connection=connection, model=model):
                        prompt = {
                            "source": {"class_type": "TestHarnessImageSource", "inputs": {}},
                            "reply": {"class_type": "llm_remote_reply", "inputs": {
                                **DEFAULTS, "connection": connection, "model": model, "images.图片1": ["source", 0]}},
                            "out": {"class_type": "TestHarnessReplySink", "inputs": {"text": ["reply", 0]}},
                        }
                        validation = asyncio.run(execution.validate_prompt("model-selection-test", prompt, None))
                        self.assertTrue(validation[0], validation)
                        self.requests.clear()
                        executor = execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                            execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
                        executor.execute(prompt, "model-selection-test", execute_outputs=["out"])
                        self.assertTrue(executor.success, executor.status_messages)
                        self.assertEqual(executor.history_result["outputs"]["out"]["text"], ["reply-1"])
                        self.assertEqual(len(self.requests), 1)
                        self.assertEqual(self.requests[0][0]["model"], model)
                        self.assertEqual(decoded_images(self.requests[0]), [((3, 250), (255, 0, 0))])

            prompt["reply"]["inputs"]["model"] = "unconfigured-model"
            validation = asyncio.run(execution.validate_prompt("model-selection-test", prompt, None))
            self.assertFalse(validation[0])
            self.assertTrue(any(error["type"] == "value_not_in_list"
                                for error in validation[3]["reply"]["errors"]))

    def test_multiple_dynamic_inputs_order_and_same_request(self):
        self.run_node(**{"images.图片10": [picture([0, 0, 1])], "images.图片2": [picture([0, 1, 0])],
                         "images.图片1": [picture([1, 0, 0], batch=2)], "image_list": [picture([1, 1, 1])]})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([color for _, color in decoded_images(self.requests[0])],
                         [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)])

    def test_regular_lists_repeat_last_with_entire_merged_list_in_each_request(self):
        result = self.run_node(**{"images.图片1": [picture([1, 0, 0]), picture([0, 1, 0])],
                                 "images.图片2": [picture([0, 0, 1])], "prompt": ["one", "two", "three"],
                                 "image_list": [picture([1, 1, 1]), picture([0, 0, 0])]})
        self.assertEqual(result, [["reply-1", "reply-2", "reply-3"]])
        self.assertEqual([request[0]["messages"][-1]["content"][-1]["text"] for request in self.requests],
                         ["one", "two", "three"])
        for request, first in zip(self.requests, [(255, 0, 0), (0, 255, 0), (0, 255, 0)]):
            self.assertEqual([color for _, color in decoded_images(request)],
                             [first, (0, 0, 255), (255, 255, 255), (0, 0, 0)])

    def test_legacy_image_inputs_still_map(self):
        result = self.run_node(image_1=[picture([1, 0, 0]), picture([0, 1, 0])],
                               image_2=[picture([0, 0, 1])], image_3=[picture([1, 1, 1])])
        self.assertEqual(result, [["reply-1", "reply-2"]])
        self.assertEqual(len(decoded_images(self.requests[0])), 3)
        self.assertEqual(decoded_images(self.requests[1])[0][1], (0, 255, 0))

    def test_text_only_and_empty_merge_are_valid(self):
        self.assertEqual(self.run_node(), [["reply-1"]])
        self.assertEqual(self.run_node(image_list=[]), [["reply-1"]])
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(all(decoded_images(request) == [] for request in self.requests))

    def test_empty_mapped_input_sends_nothing(self):
        self.assertEqual(self.run_node(**{"images.图片1": []}), [[]])
        self.assertEqual(self.requests, [])

    def test_original_dimensions_in_both_groups_and_all_formats(self):
        for fmt in ("Chat Completions", "Responses", "Anthropic Messages"):
            with self.subTest(format=fmt):
                self.config["api_format"] = fmt
                self.requests.clear()
                self.run_node(**{"images.图片1": [picture([1, 0, 0], 2000, 100)],
                                 "image_list": [picture([0, 1, 0], 80, 10000)]})
                self.assertEqual([size for size, _ in decoded_images(self.requests[0])],
                                 [(2000, 100), (80, 10000)])

    def test_removed_resize_setting_does_not_resize_or_expand_requests(self):
        self.run_node(**{"images.图片1": [picture([1, 0, 0], 2000, 100)],
                         "image_list": [picture([0, 1, 0], 80, 10000)], "max_image_side": [100, 200]})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([size for size, _ in decoded_images(self.requests[0])],
                         [(2000, 100), (80, 10000)])

    def test_old_frontend_widgets_do_not_become_inputs_or_repeat_requests(self):
        self.run_node(**{"＋ 添加 / 编辑连接": [None], "stream_output": ["old preview", "stale preview"],
                         "image_list": [picture([1, 0, 0], 5, 7), picture([0, 1, 0], 4, 2)]})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(decoded_images(self.requests[0]), [((5, 7), (255, 0, 0)), ((4, 2), (0, 255, 0))])
        self.assertNotIn("old preview", str(self.requests[0][0]))

    def test_unknown_input_still_reports_a_wiring_error(self):
        with self.assertRaisesRegex(ValueError, "未知输入口：image_typo"):
            self.run_node(image_typo=[picture([1, 0, 0])])
        self.assertEqual(self.requests, [])

    def test_connection_changes_invalidate_cached_replies(self):
        prompt = {
            "reply": {"class_type": "llm_remote_reply", "inputs": dict(DEFAULTS)},
            "out": {"class_type": "TestHarnessReplySink", "inputs": {"text": ["reply", 0]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"llm_remote_reply": reply.LLMRemoteReply, "TestHarnessReplySink": ReplySink}):
            executor = execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
            for run in range(2):
                executor.execute(deepcopy(prompt), f"cached-{run}", execute_outputs=["out"])
                self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(len(self.requests), 1, "Unchanged requests must not contact the provider again")
            for count, (key, value) in enumerate((("endpoint", "https://new.invalid/v1"), ("api_key", "new-key"),
                                                  ("api_format", "Responses")), start=2):
                self.config[key] = value
                executor.execute(deepcopy(prompt), f"changed-{count}", execute_outputs=["out"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(len(self.requests), count, "Changing a connection must invalidate the old provider's reply")

    def test_same_content_reuses_reply_across_nodes_and_copied_image_tensors(self):
        image = picture([1, 0, 0])
        first = self.run_node(node_id="loop-1:reply", image_list=[image])
        second = self.run_node(node_id="loop-2:reply", image_list=[image.clone()])
        self.assertEqual(second, first)
        self.assertEqual(len(self.requests), 1)

    def test_all_previous_inputs_remain_cached_until_restart(self):
        first = self.run_node(prompt=["A"])
        second = self.run_node(prompt=["B"])
        with patch.object(reply, "_collect_images", side_effect=AssertionError("Cache hit must skip PNG encoding")):
            self.assertEqual(self.run_node(prompt=["A"]), first)
            self.assertEqual(self.run_node(prompt=["B"]), second)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(generation_cache.CACHE._results), 2)

    def test_duplicate_list_items_reuse_reply_and_preserve_output_order(self):
        result = self.run_node(prompt=["A", "B", "A"])
        self.assertEqual(result, [["reply-1", "reply-2", "reply-1"]])
        self.assertEqual(self.run_node(prompt=["B", "A"]), [["reply-2", "reply-1"]])
        self.assertEqual(len(self.requests), 2)

    def test_each_image_port_and_merged_list_participate_in_cache_identity(self):
        red, green, blue = picture([1, 0, 0]), picture([0, 1, 0]), picture([0, 0, 1])
        original = {"images.图片1": [red], "images.图片2": [green], "image_list": [blue]}
        first = self.run_node(**original)
        for count, change in enumerate(({"images.图片1": [blue]}, {"images.图片2": [blue]},
                                       {"image_list": [green]}), start=2):
            self.run_node(**{**original, **change})
            self.assertEqual(len(self.requests), count)
        self.assertEqual(self.run_node(**dict(reversed(list(original.items())))), first)
        self.assertEqual(len(self.requests), 4)
        # Identical transmitted images but different input-port grouping is a different input.
        self.run_node(**{"images.图片1": [red], "image_list": [green, blue]})
        self.assertEqual(len(self.requests), 5)

    def test_image_hash_uses_exact_values_shape_dtype_and_batch_order(self):
        image = picture([0.5, 0, 0]).clone()
        first = self.run_node(image_list=[image])
        noncontiguous = image.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertEqual(self.run_node(image_list=[noncontiguous]), first)
        changed = image.clone()
        changed[0, 0, 0, 0] += 0.0001
        self.assertEqual(reply._tensor_to_png_b64(image), reply._tensor_to_png_b64(changed))
        for count, value in enumerate((changed, image.double(), image.reshape(1, 3, 2, 3),
                                       image.expand(2, -1, -1, -1)), start=2):
            self.run_node(image_list=[value])
            self.assertEqual(len(self.requests), count)
        batch = torch.cat([image, picture([0, 1, 0])])
        self.run_node(image_list=[batch])
        self.run_node(image_list=[batch.flip(0)])
        self.assertEqual(len(self.requests), 7)
        image[0, 0, 0, 1] = 0.1
        self.run_node(image_list=[image])
        self.assertEqual(len(self.requests), 8, "In-place mutations must invalidate the cache")

    def test_changed_text_parameters_and_image_order_make_separate_requests(self):
        red, green = picture([1, 0, 0]), picture([0, 1, 0])
        self.run_node(image_list=[red, green])
        for count, change in enumerate(({"prompt": ["different"]}, {"system_prompt": ["different"]},
                                       {"model": ["another-model"]}, {"temperature": [1.0]},
                                       {"thinking": ["高"]}, {"max_tokens": [200]}, {"timeout": [60]},
                                       {"image_list": [green, red]}, {"image_list": [red]}), start=2):
            self.run_node(**{"image_list": [red, green], **change})
            self.assertEqual(len(self.requests), count)

    def test_force_refresh_replaces_cache_and_failure_keeps_previous_reply(self):
        self.assertEqual(self.run_node(), [["reply-1"]])
        self.assertEqual(self.run_node(force_refresh=[True]), [["reply-2"]])
        self.assertEqual(self.run_node(), [["reply-2"]])
        self.assertEqual(len(self.requests), 2)
        with patch.object(reply, "_post_stream", side_effect=RuntimeError("provider unavailable")):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                self.run_node(force_refresh=[True])
            self.assertEqual(self.run_node(), [["reply-2"]])

    def test_failed_request_is_not_cached(self):
        with patch.object(reply, "_post_stream", side_effect=RuntimeError("provider unavailable")):
            with self.assertRaises(RuntimeError):
                self.run_node()
        self.assertEqual(list(Path(self.temp).rglob("*.json")), [])
        self.assertEqual(generation_cache.CACHE._results, {})
        self.assertEqual(self.run_node(), [["reply-1"]])

    def test_interrupted_request_is_not_cached(self):
        with patch.object(generation_cache, "throw_exception_if_processing_interrupted") as check:
            def respond_and_interrupt(*args, **kwargs):
                check.side_effect = InterruptProcessingException()
                return {"choices": [{"message": {"content": "interrupted reply"}}]}

            with patch.object(reply, "_post_stream", side_effect=respond_and_interrupt):
                with self.assertRaises(InterruptProcessingException):
                    self.run_node()
        self.assertEqual(generation_cache.CACHE._results, {})
        self.assertEqual(self.run_node(), [["reply-1"]])

    def test_empty_successful_reply_is_reused(self):
        with patch.object(reply, "_call_llm", return_value="") as call:
            self.assertEqual(self.run_node(), [[""]])
            self.assertEqual(self.run_node(), [[""]])
        self.assertEqual(call.call_count, 1)

    def test_cache_hit_replays_preview_to_current_client(self):
        self.run_node()
        send = Mock()
        with patch.object(server.PromptServer, "instance", SimpleNamespace(client_id="client", send_sync=send)), \
             patch.object(reply, "get_executing_context", return_value=SimpleNamespace(prompt_id="cached-run")):
            self.assertEqual(self.run_node(node_id="other-node"), [["reply-1"]])
        send.assert_called_once_with("llm_harness.stream", {
            "node_id": "other-node", "prompt_id": "cached-run", "text": "reply-1", "cached": True,
        }, "client")
        self.assertEqual(len(self.requests), 1)

    def test_cache_reset_takes_effect_in_the_same_executor(self):
        prompt = {
            "reply": {"class_type": "llm_remote_reply", "inputs": dict(DEFAULTS)},
            "out": {"class_type": "TestHarnessReplySink", "inputs": {"text": ["reply", 0]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"llm_remote_reply": reply.LLMRemoteReply, "TestHarnessReplySink": ReplySink}):
            executor = execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
            for run in range(3):
                if run == 2:
                    generation_cache.CACHE = generation_cache.GenerationCache()
                executor.execute(deepcopy(prompt), f"cache-delete-{run}", execute_outputs=["out"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(len(self.requests), 1 if run < 2 else 2)

    def test_cache_keeps_only_hash_and_reply_in_memory(self):
        self.config["api_key"] = "private-test-key"
        self.run_node(prompt=["private-test-prompt"], image_list=[picture([1, 0, 0])])
        files = list(Path(self.temp).rglob("*.json"))
        self.assertEqual(files, [])
        self.assertEqual(list(generation_cache.CACHE._results.values()), ["reply-1"])
        self.assertRegex(next(iter(generation_cache.CACHE._results)), r"^[a-f0-9]{64}$")

    def test_cache_is_empty_in_a_fresh_python_process(self):
        self.run_node(image_list=[picture([1, 0, 0])])
        script = '''
import json, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[2])
from test_image_inputs import ImageInputTests, folder_paths, picture, generation_cache
assert generation_cache.CACHE._results == {}, "New processes must start without previous replies"
case = ImageInputTests()
case.setUp()
try:
    with patch.object(folder_paths, "get_user_directory", return_value=sys.argv[1]):
        output = case.run_node(image_list=[picture([1, 0, 0])])
        print(json.dumps({"output": output, "requests": len(case.requests)}))
finally:
    case.doCleanups()
'''
        results = []
        for _ in range(2):
            process = subprocess.run([sys.executable, "-s", "-c", script, self.temp, str(Path(__file__).parent)],
                                     capture_output=True, text=True, encoding="utf-8", timeout=60,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(process.returncode, 0, process.stderr)
            results.append(json.loads(process.stdout.strip().splitlines()[-1]))
        self.assertEqual(results[0], {"output": [["reply-1"]], "requests": 1})
        self.assertEqual(results[1], {"output": [["reply-1"]], "requests": 1})

    def test_split_preview_to_llm_in_real_executor_sends_once(self):
        classes = {cls.define_schema().node_id: cls for cls in
                   (ImageSource, LiuxinyuSplitLook, reply.LLMRemoteReply, ReplySink)}
        prompt = {
            "source": {"class_type": "TestHarnessImageSource", "inputs": {}},
            "split": {"class_type": "LiuxinyuSplitLook", "inputs": {
                "image": ["source", 0], "block_height": 100, "overlap": 0, "direction": "上下切分"}},
            "reply": {"class_type": "llm_remote_reply", "inputs": {**DEFAULTS, "image_list": ["split", 0]}},
            "out": {"class_type": "TestHarnessReplySink", "inputs": {"text": ["reply", 0]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, classes):
            validation = asyncio.run(execution.validate_prompt("image-input-test", prompt, None))
            self.assertTrue(validation[0], validation)
            executor = execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
            executor.execute(prompt, "image-input-test", execute_outputs=["out"])
            self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(executor.history_result["outputs"]["out"]["text"], ["reply-1"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(decoded_images(self.requests[0])), 3)


if __name__ == "__main__":
    unittest.main()
