"""Image providers, list mapping and checkpoint isolation; no paid API calls."""
import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]

import execution
import folder_paths
import torch
from PIL import Image
from server import PromptServer
from harness_test_support import connections, request, BatchCheckpoint, APIImageEdit, APIImageGenerate, generation_cache


def encoded_image(color="red"):
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ImageAPITests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(generation_cache, "CACHE", generation_cache.GenerationCache()))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.config = {
            "first": {"endpoint": "https://first.invalid/v1", "api_key": "secret-A", "models": ["image-A"]},
            "second": {"endpoint": "https://second.invalid/v1/", "api_key": "secret-B", "models": ["image-B"]},
        }
        self.enterContext(patch.object(connections, "load_connections", return_value=self.config))
        self.enterContext(patch.object(folder_paths, "get_user_directory", return_value=self.temp))
        self.enterContext(patch.object(folder_paths, "get_temp_directory", return_value=str(Path(self.temp)/"temp")))
        self.enterContext(patch.object(PromptServer, "instance", SimpleNamespace(client_id=None), create=True))
        self.post = self.enterContext(patch.object(request.requests, "post", return_value=Mock(
            status_code=200, json=lambda: {"data": [{"b64_json": encoded_image()}]})))
        self.get = self.enterContext(patch.object(request.requests, "get", side_effect=AssertionError("Unexpected download")))

    def test_generation_routes_selected_source(self):
        result = APIImageGenerate.execute("second", "image-B", "draw", "auto", "high", 60)[0]
        self.assertEqual(tuple(result.shape), (1, 3, 4, 3))
        args, kwargs = self.post.call_args
        self.assertEqual(args[0], "https://second.invalid/v1/images/generations")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret-B"})
        self.assertEqual(kwargs["json"]["model"], "image-B")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (20, 60))

    def test_edit_lists_reference_and_mask_payloads(self):
        images = {"输入图像1": torch.zeros(2, 3, 4, 3), "输入图像2": torch.ones(1, 3, 4, 3)}
        masks = torch.ones(1, 3, 4)
        values = APIImageEdit._build_batch_payloads(images, "edit", "image-A", "auto", masks, "auto", False)
        self.assertEqual([len(v["images"]) for v in values], [2, 1])
        self.assertIn("mask", values[0])
        self.assertNotIn("mask", values[1])
        fixed = APIImageEdit._build_batch_payloads(images, "edit", "image-A", "auto", None, "high", True)
        self.assertEqual([len(v["images"]) for v in fixed], [2, 2])
        self.assertEqual(fixed[0]["quality"], "high")

    def execute_edit(self, source="first", regenerate=False, node_id="test-edit", **overrides):
        raw = {"images.输入图像1": torch.zeros(2,3,4,3), "connection": source,
               "model": "image-A", "prompt": "edit", "size": "auto", "timeout_seconds": 60,
               "max_concurrency": 1, "force_regenerate": regenerate}
        raw.update(overrides)
        inputs, missing, data = execution.get_input_data(raw, APIImageEdit, node_id)
        self.assertFalse(missing)
        return asyncio.run(execution.get_output_data("test", node_id, APIImageEdit, inputs, v3_data=data))

    def test_executor_resume_and_source_switch(self):
        result = self.execute_edit()
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(tuple(result[0][0][0].shape), (2,3,4,3))
        self.assertEqual(result[1]["llm_harness_image_progress"], [{"completed":2,"total":2}])
        self.execute_edit()
        self.assertEqual(self.post.call_count, 2)
        self.execute_edit("second")
        self.assertEqual(self.post.call_count, 4)
        self.assertEqual(self.post.call_args.args[0], "https://second.invalid/v1/images/edits")
        self.execute_edit("second", True)
        self.assertEqual(self.post.call_count, 6)

    def test_checkpoint_identity_and_secret_not_written(self):
        base = dict(model="x", key="secret-A", endpoint="https://a.invalid/v1", source="a")
        first = BatchCheckpoint(generation_cache.input_key("LLMHarnessImageEdit", base))
        for changed in ({"endpoint":"https://b.invalid/v1"},{"source":"b"},{"key":"secret-B"}):
            self.assertNotEqual(first.directory, BatchCheckpoint(generation_cache.input_key("LLMHarnessImageEdit", base|changed)).directory)
        for path in Path(self.temp).rglob("*.json"):
            self.assertNotIn("secret-", path.read_text())

    def test_download_and_saved_response_do_not_resubmit(self):
        self.post.return_value = Mock(status_code=200, json=lambda:{"data":[{"url":"https://cdn.invalid/image.png"}]})
        self.get.side_effect = None
        self.get.return_value = Mock(content=base64.b64decode(encoded_image()), raise_for_status=lambda:None)
        path = Path(self.temp)/"response.json"
        request.request_image("https://a.invalid/v1","edits",{},"secret-A",60,path)
        request.request_image("https://a.invalid/v1","edits",{},"secret-A",60,path)
        self.assertEqual(self.post.call_count, 1)
        self.assertNotIn("headers", self.get.call_args.kwargs)

    def test_request_error_redacts_key_and_allows_empty_key(self):
        self.post.return_value = Mock(status_code=400,json=lambda:{"error":{"message":"bad secret-A"}})
        with self.assertRaisesRegex(RuntimeError,"REDACTED") as error:
            request.request_image("https://a.invalid/v1","edits",{},"secret-A",60)
        self.assertNotIn("secret-A",str(error.exception))
        self.post.return_value = Mock(status_code=400,json=lambda:{"error":{"message":"bad request"}})
        with self.assertRaisesRegex(RuntimeError,"bad request"):
            request.request_image("https://a.invalid/v1","edits",{},"",60)
        self.assertEqual(self.post.call_args.kwargs["headers"],{})

    def test_schema_includes_second_provider_models(self):
        schema = APIImageGenerate.define_schema()
        models = next(x for x in schema.inputs if x.id == "model")
        self.assertIn("image-B", models.options)
        self.assertEqual(schema.display_name,"API图像生成")
        self.assertEqual(APIImageEdit.define_schema().display_name,"API图片编辑")

    def test_invalid_source_fails_before_request(self):
        for endpoint in ("file:///tmp/image", "https://a.invalid/v1?key=secret", "https://user:pass@a.invalid"):
            self.config["first"]["endpoint"] = endpoint
            with self.assertRaises(ValueError):
                APIImageGenerate.execute("first","image-A","draw","auto","auto",60)
        self.post.assert_not_called()

    def test_failed_batch_retries_only_missing_images(self):
        success = self.post.return_value
        self.post.side_effect = [success, Mock(status_code=429, json=lambda: {"error": {"message": "rate limit"}})]
        images = {"images.输入图像1": torch.zeros(3, 3, 4, 3)}
        with self.assertRaisesRegex(RuntimeError, "1/3"):
            self.execute_edit(**images)
        self.assertEqual(self.post.call_count, 2, "Stop dispatching after failure")
        self.post.side_effect = None
        result = self.execute_edit(**images)
        self.assertEqual(self.post.call_count, 4, "The saved first image must not be submitted again")
        self.assertEqual(tuple(result[0][0][0].shape), (3, 3, 4, 3))

    def test_frontend_only_fields_are_filtered_by_image_nodes(self):
        self.execute_edit(**{"添加 / 编辑来源": None, "刷新来源": None, "llm_harness_total_progress": "old progress"})
        self.assertEqual(self.post.call_count, 2)
        raw = {"connection": "first", "model": "image-A", "prompt": "draw", "size": "auto", "quality": "auto",
               "timeout_seconds": 60, "添加 / 编辑来源": None, "刷新来源": None}
        inputs, missing, data = execution.get_input_data(raw, APIImageGenerate, "test-generate")
        self.assertFalse(missing)
        asyncio.run(execution.get_output_data("test", "test-generate", APIImageGenerate, inputs, v3_data=data))
        self.assertEqual(self.post.call_count, 3)

    def test_failed_image_download_resumes_from_saved_response(self):
        self.post.return_value = Mock(status_code=200, json=lambda: {"data": [{"url": "https://cdn.invalid/image.png"}]})
        path = Path(self.temp) / "response.json"
        with self.assertRaises(RuntimeError):
            request.request_image("https://a.invalid/v1", "edits", {}, "secret-A", 60, path)
        self.get.side_effect = None
        self.get.return_value = Mock(content=base64.b64decode(encoded_image()), raise_for_status=lambda: None)
        image = request.request_image("https://a.invalid/v1", "edits", {}, "secret-A", 60, path)
        self.assertEqual(image.size, (4, 3))
        self.assertEqual(self.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
