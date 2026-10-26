import numpy as np
import torch
from comfy_api.latest import io

from .connections import resolve_connection, source_inputs
from .request import request_image


class APIImageGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMHarnessImageGenerate", display_name="API图像生成", category="llm_harness/image",
            description="从来源管理中选择接口、Key和模型，调用 /images/generations 生成一张图。来源与 LLM 远程回复共用，Key 不写入工作流。",
            not_idempotent=True,
            inputs=source_inputs() + [
                io.String.Input("prompt", display_name="提示词", default="", multiline=True),
                io.Combo.Input("size", display_name="尺寸", options=["1024x1024", "1536x1024", "1024x1536", "auto"], default="1024x1024"),
                io.Combo.Input("quality", display_name="质量", options=["auto", "low", "medium", "high"], default="auto"),
                io.Int.Input("timeout_seconds", display_name="超时（秒）", default=300, min=30, max=900),
            ], outputs=[io.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, connection, model, prompt, size, quality, timeout_seconds):
        if not prompt.strip():
            raise ValueError("请填写图片提示词。")
        endpoint, key = resolve_connection(connection, model)
        payload = {"model": model.strip(), "prompt": prompt, "n": 1, "size": size}
        if quality != "auto":
            payload["quality"] = quality
        image = request_image(endpoint, "generations", payload, key, timeout_seconds)
        return io.NodeOutput(torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0))
