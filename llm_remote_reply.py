"""LLM 远程回复 - 调用远程 LLM API 的 ComfyUI 节点（支持多图、多模型、流式返回）。"""
import base64
import io as _io
import json
import os
import subprocess
import sys

try:
    from aiohttp import web
    from server import PromptServer
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False

from comfy_api.latest import IO
from comfy.model_management import throw_exception_if_processing_interrupted


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "data")
_CONNECTIONS_FILE = os.path.join(_DATA_DIR, "connections.json")
_DIALOG_SCRIPT = os.path.join(_HERE, "scripts", "harness_dialog.py")

_THINKING_MAP = {"关闭": "off", "低": "low", "中": "medium", "高": "high"}


def _load_connections():
    if not os.path.isfile(_CONNECTIONS_FILE):
        return {}
    try:
        with open(_CONNECTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _setup_routes():
    if not _HAS_SERVER:
        return
    routes = PromptServer.instance.routes

    @routes.get("/llm_harness/connections")
    async def _list(request):
        data = _load_connections()
        public = {}
        for name, cfg in data.items():
            pub = {k: v for k, v in cfg.items() if k != "api_key"}
            if "models" not in pub and "model" in pub:
                pub["models"] = [pub["model"]]
            public[name] = pub
        return web.json_response({"connections": public})

    @routes.post("/llm_harness/connections/add")
    async def _add(request):
        try:
            flags = 0
            if sys.platform == "win32":
                flags = 0x08000000  # CREATE_NO_WINDOW
            subprocess.Popen(
                [sys.executable, "-s", _DIALOG_SCRIPT],
                cwd=_HERE,
                creationflags=flags,
            )
            return web.json_response({"ok": True, "msg": "已打开连接管理窗口"})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)


_setup_routes()


def _tensor_to_png_b64(image_tensor):
    import numpy as np
    from PIL import Image as PILImage

    if image_tensor is None:
        return []
    arr = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else np.asarray(image_tensor)
    if arr.ndim == 3:
        arr = arr[None, ...]
    out = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.ndim != 3:
            continue
        if frame.shape[-1] not in (1, 3, 4):
            continue
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = np.clip(frame, 0.0, 1.0)
        img = PILImage.fromarray((frame * 255.0).round().astype(np.uint8), mode="RGB")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _collect_images(images):
    b64_list = []
    for t in images:
        if t is None:
            continue
        throw_exception_if_processing_interrupted()
        b64_list.extend(_tensor_to_png_b64(t))
    return b64_list


def _post_stream(url, headers, body, timeout, fmt, node_id=None):
    import urllib.request
    import urllib.error

    if "User-Agent" not in headers:
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    body["stream"] = True
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content_type = r.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raw = r.read().decode("utf-8", errors="replace")
                return json.loads(raw)

            full_text = []
            for line in r:
                line = line.decode("utf-8", errors="replace").strip()
                if not line: continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = ""
                        if fmt == "Chat Completions":
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}).get("content", "")
                        elif fmt == "Responses":
                            if chunk.get("type") == "response.output_text.delta":
                                delta = chunk.get("delta", "")
                        elif fmt == "Anthropic Messages":
                            if chunk.get("type") == "content_block_delta":
                                c = chunk.get("delta", {})
                                if c.get("type") == "text_delta":
                                    delta = c.get("text", "")

                        if delta:
                            full_text.append(delta)
                            if _HAS_SERVER and node_id:
                                # 推送到前端
                                PromptServer.instance.send_sync("llm_harness.stream", {
                                    "node_id": node_id,
                                    "text": delta
                                })
                    except json.JSONDecodeError:
                        pass

            if fmt == "Chat Completions":
                return {"choices": [{"message": {"content": "".join(full_text)}}]}
            elif fmt == "Responses":
                return {"output_text": "".join(full_text), "output": []}
            elif fmt == "Anthropic Messages":
                return {"content": [{"type": "text", "text": "".join(full_text)}]}

    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} 来自 {url}：{raw[:1000]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"连接失败 {url}：{e.reason}")


def _call_llm(cfg, model, prompt, system_prompt, thinking, temperature, max_tokens, timeout, images_b64, node_id=None):
    fmt = cfg.get("api_format", "Chat Completions")
    endpoint = (cfg.get("endpoint") or "").rstrip("/")
    api_key = cfg.get("api_key") or ""
    if not endpoint:
        raise ValueError("连接配置缺少接口地址")
    if not model:
        raise ValueError("连接配置缺少模型 ID")

    effort = _THINKING_MAP.get(thinking, thinking)

    if fmt == "Chat Completions":
        url = endpoint + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        user_content = []
        for b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + b64},
            })
        user_content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        body = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if effort and effort != "off":
            body["reasoning_effort"] = effort
        resp = _post_stream(url, headers, body, timeout, fmt, node_id)
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"无法解析响应：{json.dumps(resp, ensure_ascii=False)[:800]}")

    if fmt == "Responses":
        url = endpoint + "/responses"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        content_list = []
        for b64 in images_b64:
            content_list.append({
                "type": "input_image",
                "image_url": "data:image/png;base64," + b64,
            })
        content_list.append({"type": "input_text", "text": prompt})

        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": content_list
            }
        ]
        body = {
            "model": model,
            "input": input_items,
            "max_output_tokens": int(max_tokens),
        }
        if system_prompt:
            body["instructions"] = system_prompt
        if effort and effort != "off":
            body["reasoning"] = {"effort": effort}
        else:
            body["temperature"] = float(temperature)
        resp = _post_stream(url, headers, body, timeout, fmt, node_id)
        texts = []
        for item in resp.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        t = c.get("text")
                        if t:
                            texts.append(t)
        if not texts and isinstance(resp.get("output_text"), str):
            texts.append(resp["output_text"])
        if not texts:
            raise RuntimeError(f"无法解析响应：{json.dumps(resp, ensure_ascii=False)[:800]}")
        return "\n".join(texts)

    if fmt == "Anthropic Messages":
        url = endpoint + "/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key
        content_blocks = []
        for b64 in images_b64:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        content_blocks.append({"type": "text", "text": prompt})
        body = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": content_blocks}],
        }
        if system_prompt:
            body["system"] = system_prompt
        if effort and effort != "off":
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
        else:
            body["temperature"] = float(temperature)
        resp = _post_stream(url, headers, body, timeout, fmt, node_id)
        texts = []
        for block in resp.get("content", []):
            if block.get("type") == "text":
                t = block.get("text")
                if t:
                    texts.append(t)
        if not texts:
            raise RuntimeError(f"无法解析响应：{json.dumps(resp, ensure_ascii=False)[:800]}")
        return "\n".join(texts)

    raise ValueError(f"未知的接口格式：{fmt}")


class LLMRemoteReply(IO.ComfyNode):
    """调用远程 LLM（支持图片输入；Chat Completions / Responses / Anthropic Messages）。"""

    @classmethod
    def define_schema(cls):
        data = _load_connections()
        conns = sorted(data.keys())
        options = conns if conns else ["（尚未配置连接，请点击节点上的添加连接按钮）"]

        default_models = ["（未配置模型）"]
        if conns:
            first_cfg = data[conns[0]]
            models = first_cfg.get("models", [])
            if not models and first_cfg.get("model"):
                models = [first_cfg["model"]]
            if models:
                default_models = models

        return IO.Schema(
            node_id="llm_remote_reply",
            display_name="LLM 远程回复",
            category="llm_harness",
            description=(
                "调用远程 LLM 生成文本回复。普通图片输入连接后自动增加，可同时携带多张图片；"
                "普通口接执行列表时仍逐项执行。列表合并图片口会将整个执行列表及其内部批次"
                "按顺序一起发送，支持不同尺寸。每次请求先发送普通口图片，再发送合并列表。"
                "提示词等其他输入若为执行列表，仍按项执行，每次携带完整合并图片列表。"
                "图片按输入宽高编码为 PNG，不在本地缩放；超出服务端限制时由服务端处理或返回错误。"
            ),
            inputs=[
                IO.Combo.Input("connection", display_name="连接", options=options, default=options[0]),
                IO.Combo.Input("model", display_name="模型", options=default_models, default=default_models[0]),
                IO.String.Input("prompt", display_name="提示词", multiline=True, default=""),
                IO.String.Input("system_prompt", display_name="系统提示词", multiline=True, default=""),
                IO.Combo.Input("thinking", display_name="思考强度", options=list(_THINKING_MAP.keys()), default="关闭"),
                IO.Float.Input("temperature", display_name="温度", default=0.7, min=0.0, max=2.0),
                IO.Int.Input("max_tokens", display_name="最大输出长度", default=4096, min=1, max=200000),
                IO.Int.Input("timeout", display_name="超时（秒）", default=180, min=1, max=1200,
                             tooltip="网络连接及阻塞读写等待超时，不是首个 token 或整条回复的总时限。流式返回期间仍生效；持续收到数据（含心跳）时，总耗时可以超过此值。"),
                IO.Autogrow.Input("images", display_name="图片", template=IO.Autogrow.TemplateNames(
                    input=IO.Image.Input("image", tooltip="可同时连接多张图片。执行列表按项运行；整组合并请接列表合并图片。"),
                    names=[f"图片{i}" for i in range(1, 101)], min=0,
                )),
                IO.Image.Input("image_list", display_name="列表合并图片", optional=True,
                               tooltip="整个执行列表一次性提交；包括每项内部的全部批次图片，允许不同尺寸。"),
            ],
            outputs=[IO.String.Output("text", display_name="回复文本", is_output_list=True)],
            hidden=[IO.Hidden.unique_id],
            is_input_list=True,
            accept_all_inputs=True,  # 保留旧 API 工作流的 image_1 / image_2 / image_3 接线。
        )

    @classmethod
    def execute(cls, connection, model, prompt, system_prompt, thinking, temperature, max_tokens, timeout,
                images=None, image_list=None, **legacy_images):
        # Old API workflows may still send this removed setting; it no longer resizes or maps requests.
        legacy_images.pop("max_image_side", None)
        images = dict(images or {})
        for name, value in legacy_images.items():
            if name not in ("image_1", "image_2", "image_3"):
                raise ValueError(f"未知输入口：{name}")
            image_name = f"图片{name[-1]}"
            if image_name in images:
                raise ValueError(f"{image_name} 同时使用了新旧输入口，请只保留一个。")
            images[image_name] = value
        images = dict(sorted(images.items(), key=lambda item: int(item[0][2:])))

        # INPUT_IS_LIST 对所有输入生效；只有 image_list 不参与逐项配对。
        mapped = dict(connection=connection, model=model, prompt=prompt, system_prompt=system_prompt,
                      thinking=thinking, temperature=temperature, max_tokens=max_tokens,
                      timeout=timeout, **images)
        mapped = {name: value if isinstance(value, (list, tuple)) else [value]
                  for name, value in mapped.items()}
        if any(len(values) == 0 for values in mapped.values()):
            return IO.NodeOutput([])
        if image_list is None:
            image_list = []
        elif not isinstance(image_list, (list, tuple)):
            image_list = [image_list]

        connections = _load_connections()
        node_id = getattr(cls.hidden, 'unique_id', None)
        replies = []
        for index in range(max(len(values) for values in mapped.values())):
            throw_exception_if_processing_interrupted()
            current = {name: values[min(index, len(values) - 1)] for name, values in mapped.items()}
            current_images = [current.pop(name) for name in images]
            connection_name = current.pop("connection")
            cfg = connections.get(connection_name)
            if not cfg:
                raise ValueError(f"未找到连接「{connection_name}」。请点击节点上的「＋ 添加 / 编辑连接」按钮配置。")
            images_b64 = _collect_images([*current_images, *image_list])
            current["prompt"] = current["prompt"] or ""
            current["system_prompt"] = current["system_prompt"] or ""
            replies.append(_call_llm(cfg, images_b64=images_b64, node_id=node_id, **current))
        return IO.NodeOutput(replies)
