"""Exercise ComfyUI list mapping and request bodies without contacting a model."""

import asyncio
import base64
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK.parent / "comfyui-liuxinyu-packs"))

from aiohttp import web
from PIL import Image
import torch
import execution
import nodes
import server
from comfy_api.latest import IO
from split_preview_nodes import LiuxinyuSplitLook

with patch.object(server.PromptServer, "instance", SimpleNamespace(routes=web.RouteTableDef()), create=True):
    import llm_remote_reply as reply


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

    def run_node(self, **inputs):
        live_inputs = {**DEFAULTS, **dict.fromkeys(inputs)}
        _, missing, v3_data = execution.get_input_data(live_inputs, reply.LLMRemoteReply, "reply")
        self.assertFalse(missing)
        mapped = {key: [value] for key, value in DEFAULTS.items()}
        mapped.update(inputs)
        outputs, _, _, _ = asyncio.run(execution.get_output_data(
            "image-input-test", "reply", reply.LLMRemoteReply, mapped, v3_data=v3_data))
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
        self.assertEqual(self.run_node(image_list=[]), [["reply-2"]])
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
