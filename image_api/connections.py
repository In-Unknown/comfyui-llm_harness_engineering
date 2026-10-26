"""Image nodes share the remote LLM's local connection store."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from comfy_api.latest import io


CONNECTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "connections.json"


def load_connections():
    if not CONNECTIONS_FILE.is_file():
        return {}
    return json.loads(CONNECTIONS_FILE.read_text(encoding="utf-8"))


def source_inputs():
    connections = load_connections()
    names = sorted(connections) or ["（请添加来源）"]
    cfg = connections.get(names[0], {})
    default_models = cfg.get("models") or ([cfg["model"]] if cfg.get("model") else ["（未配置模型）"])
    models = list(dict.fromkeys(default_models + [model for value in connections.values()
                   for model in (value.get("models") or ([value["model"]] if value.get("model") else []))]))
    return [
        io.Combo.Input("connection", display_name="来源", options=names, default=names[0]),
        io.Combo.Input("model", display_name="模型", options=models, default=models[0]),
    ]


def resolve_connection(name, model):
    cfg = load_connections().get(name)
    if not cfg:
        raise ValueError(f"未找到来源「{name}」，请点击添加 / 编辑来源配置。")
    endpoint = (cfg.get("endpoint") or "").strip().rstrip("/")
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("来源 Endpoint 必须是 HTTP(S) API 基础地址，例如 https://example.com/v1，不含查询参数或账号密码。")
    if not model or model.startswith("（"):
        raise ValueError("请在来源中配置图像模型并选择。")
    return endpoint, cfg.get("api_key") or ""


def redact(message, key):
    return str(message).replace(key, "[REDACTED]") if key else str(message)
