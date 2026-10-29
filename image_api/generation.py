import numpy as np
import torch
from comfy_api.latest import io

from .. import generation_cache
from .connections import resolve_connection, source_inputs
from .request import request_image


class APIImageGenerate(generation_cache.CachedGenerationNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMHarnessImageGenerate", display_name="API图像生成", category="llm_harness/image",
            description="从来源管理中选择接口、Key和模型，调用 /images/generations 生成一张图。来源与 LLM 远程回复共用，Key 不写入工作流。全部输入相同时复用本次 ComfyUI 启动以来的结果；重启清空。开启重新生成可更新这组输入的结果。",
            not_idempotent=True,
            inputs=source_inputs() + [
                io.String.Input("prompt", display_name="提示词", default="", multiline=True),
                io.Combo.Input("size", display_name="尺寸", options=["1024x1024", "1536x1024", "1024x1536", "auto"], default="1024x1024"),
                io.Combo.Input("quality", display_name="质量", options=["auto", "low", "medium", "high"], default="auto"),
                io.Int.Input("timeout_seconds", display_name="超时（秒）", default=300, min=30, max=900),
                io.Boolean.Input("force_regenerate", display_name="重新生成", default=False, optional=True,
                                 tooltip="关闭时复用本次启动以来相同输入的图片；开启时重新请求，成功后更新缓存。"),
            ], outputs=[io.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, connection, model, prompt, size, quality, timeout_seconds, force_regenerate=False):
        if not prompt.strip():
            raise ValueError("请填写图片提示词。")
        endpoint, key = resolve_connection(connection, model)
        payload = {"model": model.strip(), "prompt": prompt, "n": 1, "size": size}
        if quality != "auto":
            payload["quality"] = quality
        cache_key = generation_cache.input_key("LLMHarnessImageGenerate", {
            "connection": connection, "endpoint": endpoint, "api_key": key,
            "model": model, "prompt": prompt, "size": size, "quality": quality,
            "timeout_seconds": timeout_seconds,
        })
        image, _ = generation_cache.cached_image(
            cache_key, lambda: request_image(endpoint, "generations", payload, key, timeout_seconds),
            force_refresh=force_regenerate,
        )
        return io.NodeOutput(torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0))
