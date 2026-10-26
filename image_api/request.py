"""JSON image API transport; generated image downloads never receive the API key."""

import base64
import json
from io import BytesIO

import requests
from PIL import Image, ImageOps

from .checkpoint import write_json
from .connections import redact


def request_image(endpoint, operation, payload, key, timeout_seconds, response_path=None):
    if response_path is not None and response_path.exists():
        item = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        headers = {"Authorization": "Bearer " + key} if key else {}
        try:
            response = requests.post(endpoint + "/images/" + operation, headers=headers,
                                     json=payload, timeout=(20, timeout_seconds), allow_redirects=False)
        except requests.Timeout:
            raise RuntimeError("图片请求超时，服务端可能已生成或计费；请先检查来源记录再重试。") from None
        except requests.RequestException:
            raise RuntimeError("无法连接图片接口，请检查来源地址及网络。") from None
        if not 200 <= response.status_code < 300:
            try:
                error = response.json().get("error", {})
                message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                message = "非 JSON 错误响应"
            raise RuntimeError(f"图片接口 HTTP {response.status_code}: {redact(message, key)[:600]}")
        try:
            item = response.json()["data"][0]
            if not isinstance(item, dict) or not (item.get("b64_json") or item.get("url")):
                raise ValueError("No image")
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError("接口没有返回 data 图片列表或有效图片字段，请确认所选模型支持图像接口。") from None
        if response_path is not None:
            write_json(response_path, item)
    try:
        if item.get("b64_json"):
            content = base64.b64decode(item["b64_json"], validate=True)
        elif item.get("url", "").startswith("https://"):
            download = requests.get(item["url"], timeout=(20, 90))
            download.raise_for_status()
            content = download.content
        else:
            raise ValueError("Missing image or non-HTTPS image URL")
        with Image.open(BytesIO(content)) as image:
            result = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        raise RuntimeError("已收到图片响应，但下载或解码失败；请检查来源记录，避免重复生成计费。") from None
    return result
