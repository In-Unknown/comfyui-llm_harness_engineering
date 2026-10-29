"""Session reuse across all generators, including concurrent work and process restarts."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from harness_test_support import pack, generation_cache, APIImageGenerate, APIImageEdit
import test_image_api as image_tests

import execution
import folder_paths
import nodes
import torch
from comfy_api.latest import io
from comfy.model_management import InterruptProcessingException

encoded_image = image_tests.encoded_image


class ImageSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CacheTestImageSource", inputs=[], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.zeros(1, 3, 4, 3))


class ImageSink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CacheTestImageSink", inputs=[io.Image.Input("image")], outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, image):
        return io.NodeOutput(ui={"pixel": [image[0, 0, 0].tolist()]})


class GenerationCacheTests(unittest.TestCase):
    def setUp(self):
        self.api = image_tests.ImageAPITests()
        self.api.setUp()
        self.addCleanup(self.api.doCleanups)

    def generate(self, **overrides):
        args = dict(connection="first", model="image-A", prompt="draw", size="auto", quality="auto", timeout_seconds=60)
        return APIImageGenerate.execute(**(args | overrides))[0]

    def test_all_registered_generators_share_the_foundation(self):
        self.assertEqual(set(pack.NODE_CLASS_MAPPINGS), {"llm_remote_reply", "LLMHarnessImageEdit", "LLMHarnessImageGenerate"})
        for node in pack.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(issubclass(node, generation_cache.CachedGenerationNode))
            module = sys.modules[node.__module__]
            self.assertIs(module.generation_cache, generation_cache)

    def test_generation_history_and_output_mutation_isolation(self):
        original = self.generate()
        expected = original.clone()
        original.zero_()
        self.generate(prompt="another")
        self.assertTrue(torch.equal(self.generate(), expected))
        self.assertEqual(self.api.post.call_count, 2)
        self.assertTrue(all(isinstance(value, bytes) for value in generation_cache.CACHE._results.values()))

    def test_every_generation_parameter_and_connection_change_invalidates(self):
        self.generate()
        for count, change in enumerate(({"model": "image-B"}, {"prompt": "different"}, {"size": "1024x1024"},
                                       {"quality": "high"}, {"timeout_seconds": 90}, {"connection": "second"}), 2):
            self.generate(**change)
            self.assertEqual(self.api.post.call_count, count)
        for count, (field, value) in enumerate((("endpoint", "https://changed.invalid/v1"), ("api_key", "new-key")), 8):
            self.api.config["first"][field] = value
            self.generate()
            self.assertEqual(self.api.post.call_count, count)

    def test_edit_history_reuses_across_nodes_without_reencoding(self):
        first = self.api.execute_edit()
        self.api.execute_edit(prompt="different")
        with patch.object(APIImageEdit, "_build_batch_payloads", side_effect=AssertionError("Must skip input encoding")):
            again = self.api.execute_edit(node_id="copied-node")
        self.assertTrue(torch.equal(first[0][0][0], again[0][0][0]))
        self.assertEqual(self.api.post.call_count, 4)
        self.assertEqual(again[1]["llm_harness_image_progress"], [{"completed": 2, "total": 2}])

    def test_edit_ports_masks_exact_pixels_and_parameters_invalidate(self):
        main = torch.zeros(2, 3, 4, 3)
        reference = torch.ones(2, 3, 4, 3)
        mask = torch.zeros(2, 3, 4)
        base = {"images.输入图像1": main, "images.输入图像2": reference, "known_mask": mask}
        self.api.execute_edit(**base)
        changes = ({"images.输入图像1": main + 0.0001}, {"images.输入图像2": reference - 0.0001},
                   {"known_mask": mask + 0.0001}, {"known_mask": mask.double()},
                   {"prompt": "different"}, {"model": "image-B"}, {"size": "1024x1024"},
                   {"quality": "high"}, {"timeout_seconds": 90}, {"fixed_reference": True}, {"max_concurrency": 2})
        for count, change in enumerate(changes, 2):
            self.api.execute_edit(**(base | change))
            self.assertEqual(self.api.post.call_count, count * 2)
        self.api.execute_edit(**base)
        self.assertEqual(self.api.post.call_count, (len(changes) + 1) * 2)

    def test_cached_edit_restores_deleted_previews_and_ignores_mutated_output(self):
        first = self.api.execute_edit()
        expected = first[0][0][0].clone()
        first[0][0][0].zero_()
        for path in Path(self.api.temp).rglob("*.png"):
            path.unlink()
        again = self.api.execute_edit()
        self.assertTrue(torch.equal(again[0][0][0], expected))
        for preview in again[1]["images"]:
            path = Path(folder_paths.get_temp_directory()) / preview["subfolder"] / preview["filename"]
            self.assertTrue(path.is_file())
        self.assertEqual(self.api.post.call_count, 2)

    def test_force_regenerate_updates_images_but_failure_keeps_successes(self):
        for edit in (False, True):
            with self.subTest(edit=edit):
                run = (lambda force=False: self.api.execute_edit(regenerate=force)[0][0][0]) if edit else \
                      (lambda force=False: self.generate(force_regenerate=force))
                self.api.post.return_value = Mock(status_code=200, json=lambda: {"data": [{"b64_json": encoded_image("red")}]})
                original = run()
                self.api.post.return_value = Mock(status_code=200, json=lambda: {"data": [{"b64_json": encoded_image("blue")}]})
                refreshed = run(True)
                self.assertFalse(torch.equal(original, refreshed))
                self.assertTrue(torch.equal(run(), refreshed))
                self.api.post.side_effect = RuntimeError("unavailable")
                try:
                    with self.assertRaisesRegex(RuntimeError, "unavailable"):
                        run(True)
                    self.assertTrue(torch.equal(run(), refreshed))
                finally:
                    self.api.post.side_effect = None

    def test_same_inputs_are_single_flight_but_distinct_inputs_run_in_parallel(self):
        cache = generation_cache.CACHE
        entered, release = Event(), Event()
        start = Barrier(4)

        def compute():
            entered.set()
            self.assertTrue(release.wait(5))
            return "answer"

        generate = Mock(side_effect=compute)

        def run():
            start.wait(timeout=5)
            return cache.run("same", generate)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run) for _ in range(4)]
            try:
                self.assertTrue(entered.wait(5))
            finally:
                release.set()
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(sorted(reused for _, reused in results), [False, True, True, True])
        parallel = Barrier(2)

        def concurrent():
            parallel.wait(timeout=5)
            return "parallel"

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(cache.run, "first", concurrent)
            second = pool.submit(cache.run, "second", concurrent)
            self.assertEqual(first.result(timeout=5), ("parallel", False))
            self.assertEqual(second.result(timeout=5), ("parallel", False))

    def test_cancellation_or_failure_does_not_commit_or_hold_the_key_lock(self):
        cache = generation_cache.CACHE
        cache.run("existing", lambda: "previous")
        for key in ("existing", "new"):
            with patch.object(generation_cache, "throw_exception_if_processing_interrupted") as check:
                def cancel():
                    check.side_effect = InterruptProcessingException()
                    return "incomplete"

                with self.assertRaises(InterruptProcessingException):
                    cache.run(key, cancel, force_refresh=True)
            self.assertEqual(cache.get(key), "previous" if key == "existing" else None)
        self.assertEqual(cache.run("new", lambda: "complete"), ("complete", False))

    def test_concurrent_image_nodes_reuse_results_and_write_valid_previews(self):
        start = Barrier(4)

        def run(edit):
            start.wait(timeout=5)
            return self.api.execute_edit()[0][0][0] if edit else self.generate()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, edit) for edit in (False, False, True, True)]
            images = [future.result(timeout=10) for future in futures]
        self.assertTrue(torch.equal(images[0], images[1]))
        self.assertTrue(torch.equal(images[2], images[3]))
        self.assertEqual(self.api.post.call_count, 3)

    def test_real_executor_checks_connection_force_refresh_and_session_reset(self):
        classes = dict(pack.NODE_CLASS_MAPPINGS)
        classes.update(CacheTestImageSource=ImageSource, CacheTestImageSink=ImageSink)
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, classes):
            for node_type in ("LLMHarnessImageGenerate", "LLMHarnessImageEdit"):
                with self.subTest(node_type=node_type):
                    inputs = dict(connection="first", model="image-A", prompt="draw", size="auto", quality="auto", timeout_seconds=60)
                    if node_type == "LLMHarnessImageEdit":
                        inputs["images.输入图像1"] = ["source", 0]
                    graph = {
                        "source": {"class_type": "CacheTestImageSource", "inputs": {}},
                        "generate": {"class_type": node_type, "inputs": inputs},
                        "out": {"class_type": "CacheTestImageSink", "inputs": {"image": ["generate", 0]}},
                    }
                    executor = execution.PromptExecutor(SimpleNamespace(client_id=None, send_sync=lambda *args: None),
                                                        execution.CacheType.RAM_PRESSURE, {"ram": 0, "ram_inactive": 0})
                    start = self.api.post.call_count
                    for index, expected in enumerate((1, 1, 2, 3, 4, 4, 5)):
                        if index == 2:
                            self.api.config["first"]["api_key"] += "-changed"
                        inputs["force_regenerate"] = index in (3, 4)
                        if index == 6:
                            generation_cache.CACHE = generation_cache.GenerationCache()
                        executor.execute(deepcopy(graph), f"cache-{node_type}-{index}", execute_outputs=["out"])
                        self.assertTrue(executor.success, executor.status_messages)
                        self.assertEqual(self.api.post.call_count - start, expected)

    def test_process_restart_ignores_previous_image_backups(self):
        self.generate()
        self.api.execute_edit()
        old_session = generation_cache.CACHE.session_id
        script = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from test_image_api import ImageAPITests
from harness_test_support import generation_cache, APIImageGenerate
import folder_paths
assert generation_cache.CACHE.session_id != sys.argv[3]
assert generation_cache.CACHE._results == {}
case = ImageAPITests()
case.setUp()
try:
    with patch.object(folder_paths, "get_user_directory", return_value=sys.argv[2]), patch.object(folder_paths, "get_temp_directory", return_value=str(Path(sys.argv[2]) / "temp")):
        APIImageGenerate.execute("first", "image-A", "draw", "auto", "auto", 60)
        case.execute_edit()
        print(json.dumps({"requests": case.post.call_count}))
finally:
    case.doCleanups()
'''
        result = subprocess.run([sys.executable, "-s", "-c", script, str(Path(__file__).parent), self.api.temp, old_session],
                                capture_output=True, text=True, encoding="utf-8", timeout=60,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), {"requests": 3})


if __name__ == "__main__":
    unittest.main()
