import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from threading import Lock

import numpy as np
import torch
from PIL import Image, ImageOps
from comfy_api.latest import io
from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy_execution.utils import get_executing_context
from server import PromptServer
from .checkpoint import BatchCheckpoint
from .connections import resolve_connection, source_inputs, redact
from .request import request_image


class APIImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"输入图像{i}" for i in range(1, 101)],
            min=1,
        )
        return io.Schema(
            node_id="LLMHarnessImageEdit", display_name="API图片编辑",
            category="llm_harness/image", not_idempotent=True, has_intermediate_output=True, is_input_list=True,
            description="从来源管理选择接口、Key 和模型，通过 /images/edits 的 JSON 图片协议编辑图片。图片输入口随连接自动增加，最多 100 路（实际支持数量以接口为准）。输入图像1是待编辑主图，其余图片的用途请在提示词中说明。主图批次逐张编辑并按顺序输出，显示完成比例。成功图片立即保存到 Temp 和 user/llm_harness_image_resume；相同来源及输入自动复用，失败后只续做缺失项。开启重新生成可创建新一轮结果并保留旧备份。返回尺寸不一致时等比缩放并补黑边到首张尺寸。",
            inputs=source_inputs() + [
                io.Autogrow.Input(
                    "images", template=image_template,
                    tooltip="按输入图像1、2、3…顺序发送。图1为主图；其余图片没有预设用途，请在提示词中说明。跳过未连接或当前批次缺少的图片后，请求内图片编号会依次前移。",
                ),
                io.String.Input("prompt", default="只修改指定部件，其余内容完全保留。", multiline=True),
                io.Combo.Input("size", options=["1024x1024", "auto", "1536x1024", "1024x1536", "2048x2048", "3072x2688"], default="1024x1024"),
                io.Int.Input("timeout_seconds", default=300, min=30, max=900),
                io.Mask.Input("known_mask", optional=True, tooltip="蒙版遮罩定义区域信息，用于引导模型理解要修改的部分，具体修改要求请在提示词中说明。对应输入图像1，尺寸须一致。按主图批次序号匹配，缺少时不传蒙版；固定参考不影响蒙版。"),
                io.Combo.Input("quality", options=["auto", "low", "medium", "high"], default="auto", optional=True),
                io.Boolean.Input("fixed_reference", display_name="固定参考", default=False, optional=True, tooltip="开启：输入图像1的每张主图共用其余输入口各自的第一张图片。关闭：按批次序号配对，图片不足的位置不发送该输入口的图片。"),
                io.Int.Input("max_concurrency", display_name="并发请求数", default=2, min=1, max=8, optional=True, tooltip="同时发送的独立编辑请求数；每张主图请求一次并分别计费，遇到限流可设为 1。"),
                io.Boolean.Input("force_regenerate", display_name="重新生成", default=False, optional=True, tooltip="关闭：输入不变时复用已完成的图片，只续做缺失项，包括重启后。开启：每次运行新建一轮生成，不使用旧结果。续做前请关闭。"),
            ], outputs=[io.Image.Output(display_name="IMAGE", is_output_list=True)], hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def _build_batch_payloads(cls, images, prompt, model, size, known_mask, quality, fixed_reference):
        if not prompt.strip():
            raise RuntimeError("请填写编辑要求。")
        image = images.get("输入图像1")
        if image is None:
            raise ValueError("请连接输入图像1作为待编辑主图。")
        if len(image) == 0:
            raise ValueError("主图批次不能为空。")
        references = [
            batch for name, batch in sorted(images.items(), key=lambda item: int(item[0][len("输入图像"):]))
            if name != "输入图像1" and batch is not None
        ]
        payloads = []
        for index, main_image in enumerate(image):
            ref_index = 0 if fixed_reference else index
            request_images = [main_image] + [batch[ref_index] for batch in references if ref_index < len(batch)]
            mask = known_mask[index] if known_mask is not None and index < len(known_mask) else None
            payloads.append(cls._build_payload(request_images, prompt, model, size, mask, quality))
        return payloads

    @classmethod
    def execute(cls, images, prompt, model, size, timeout_seconds, connection, known_mask=None, quality="auto", fixed_reference=False, max_concurrency=2, force_regenerate=False):
        # Match ComfyUI's repeat-last list mapping, but own the entire run so
        # progress has a fixed denominator across all prompt/image batches.
        values = dict(connection=connection, prompt=prompt, model=model, size=size, timeout_seconds=timeout_seconds,
                      known_mask=known_mask, quality=quality, fixed_reference=fixed_reference,
                      max_concurrency=max_concurrency, force_regenerate=force_regenerate)
        values = {name: value if isinstance(value, list) else [value] for name, value in values.items()}
        image_values = {name: value for name, value in images.items() if value is not None}
        all_lists = list(values.values()) + list(image_values.values())
        if any(not value for value in all_lists):
            return io.NodeOutput([], ui={"images": [], "llm_harness_image_progress": [{"completed": 0, "total": 0}]})
        count = max(len(value) for value in all_lists)
        main_images = image_values.get("输入图像1")
        if not main_images:
            raise ValueError("请连接输入图像1作为待编辑主图。")
        total = sum(len(main_images[min(index, len(main_images) - 1)]) for index in range(count))
        node_id = cls.hidden.unique_id
        progress = ProgressBar(total, node_id=node_id)
        progress.update_absolute(0)
        outputs, previews = [], []
        completed = 0
        for batch_index in range(count):
            throw_exception_if_processing_interrupted()
            arguments = {name: value[min(batch_index, len(value) - 1)] for name, value in values.items()}
            batch_images = {name: value[min(batch_index, len(value) - 1)] for name, value in image_values.items()}
            source = arguments.pop("connection")
            endpoint, key = resolve_connection(source, arguments["model"])
            tensor, files = cls._execute_batch(
                batch_images, key=key, endpoint=endpoint, source=source, progress=progress, batch_index=batch_index,
                completed_before=completed, total=total, **arguments,
            )
            outputs.append(tensor)
            previews.extend(files)
            completed += len(files)
        return io.NodeOutput(outputs, ui={"images": previews,
            "llm_harness_image_progress": [{"completed": completed, "total": total}]})

    @classmethod
    def _execute_batch(cls, images, prompt, model, size, timeout_seconds, known_mask, quality,
                       fixed_reference, max_concurrency, force_regenerate, key, endpoint, source, progress,
                       batch_index, completed_before, total):
        payloads = cls._build_batch_payloads(images, prompt, model, size, known_mask, quality, fixed_reference)
        node_id = cls.hidden.unique_id if cls.hidden is not None else None
        checkpoint = BatchCheckpoint(payloads, key, node_id, force_regenerate, endpoint=endpoint, source=source)
        results = [checkpoint.load(index) for index in range(len(payloads))]
        missing = [index for index, result in enumerate(results) if result is None]
        reused = len(results) - len(missing)
        context = get_executing_context()
        server = PromptServer.instance
        client_id = server.client_id
        display_node = cls.hidden.dynprompt.get_display_node_id(node_id) if cls.hidden.dynprompt else node_id
        preview_images = {
            index: {"filename": checkpoint.image_path(index).name,
                    "subfolder": checkpoint.temp_subfolder, "type": "temp"}
            for index, result in enumerate(results) if result is not None
        }
        preview_lock = Lock()

        def publish_preview(index=None):
            # Workers publish only after the PNG is fully saved. Capture execution
            # context above: pool threads do not inherit ComfyUI's contextvars.
            with preview_lock:
                if index is not None:
                    preview_images[index] = {
                        "filename": checkpoint.image_path(index).name,
                        "subfolder": checkpoint.temp_subfolder, "type": "temp",
                    }
                if context is not None and client_id is not None:
                    server.send_sync("llm_harness_image_preview", {
                        "prompt_id": context.prompt_id,
                        "node": str(display_node),
                        "batch": batch_index,
                        "images": [preview_images[i] for i in sorted(preview_images)],
                        "completed": completed_before + len(preview_images), "total": total,
                    }, client_id)

        publish_preview()
        failures = {}
        last_status = None

        def report(active, waiting):
            nonlocal last_status
            completed = sum(result is not None for result in results)
            status = (completed, active, len(failures), waiting)
            if status == last_status:
                return
            last_status = status
            progress.update_absolute(completed_before + completed)
            checkpoint.progress(len(results), completed, active, len(failures), waiting, reused)

        def generate(index):
            result = request_image(endpoint, "edits", payloads[index], key, timeout_seconds, checkpoint.response_path(index))
            # Save inside the worker so an interrupted main loop cannot lose a success.
            checkpoint.save(index, result)
            publish_preview(index)
            return result

        report(0, len(missing))
        workers = max(1, min(int(max_concurrency), 8, len(payloads)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            next_item = 0
            while futures or (next_item < len(missing) and not failures):
                throw_exception_if_processing_interrupted()
                while not failures and next_item < len(missing) and len(futures) < workers:
                    index = missing[next_item]
                    futures[pool.submit(generate, index)] = index
                    next_item += 1
                report(len(futures), len(missing) - next_item)
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    index = futures.pop(future)
                    try:
                        results[index] = future.result()
                    except Exception as error:
                        failures[index] = redact(error, key)[:600]
                report(len(futures), len(missing) - next_item)
        if failures:
            details = "；".join(f"第 {index + 1} 张：{message}" for index, message in sorted(failures.items()))
            completed = sum(result is not None for result in results)
            raise RuntimeError(f"批次编辑未完成，已保存 {completed}/{len(results)} 张。{details}。已停止发送剩余请求。Temp：{checkpoint.temp_directory}；持久备份：{checkpoint.directory}。保持输入不变并关闭“重新生成”后重跑，只续做未完成项；超时请求可能已计费，请先核对中转站记录。")

        target_size = results[0].size
        tensors = []
        for result in results:
            if result.size != target_size:
                result = ImageOps.pad(result, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
            tensors.append(torch.from_numpy(np.array(result).astype(np.float32) / 255.0))
        return torch.stack(tensors), [preview_images[index] for index in range(len(results))]

    @classmethod
    def _build_payload(cls, images, prompt, model, size, known_mask, quality):
        image_inputs = []
        mask_input = None
        for index, tensor in enumerate(images):
            array = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            buffer = BytesIO()
            pil = Image.fromarray(array)
            if index == 0 and known_mask is not None:
                alpha = (known_mask.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                if alpha.shape != array.shape[:2]:
                    raise ValueError("known_mask must match input image dimensions")
                mask_image = Image.new("RGBA", pil.size, (255, 255, 255, 255))
                mask_image.putalpha(Image.fromarray(alpha))
                mask_buffer = BytesIO()
                mask_image.save(mask_buffer, format="PNG")
                mask_input = {"image_url": "data:image/png;base64," + base64.b64encode(mask_buffer.getvalue()).decode("ascii")}
            pil.save(buffer, format="PNG")
            image_inputs.append({"image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
        payload = {"model": model, "prompt": prompt, "n": 1, "size": size, "images": image_inputs}
        if mask_input is not None:
            payload["mask"] = mask_input
        if quality != "auto":
            payload["quality"] = quality
        return payload
