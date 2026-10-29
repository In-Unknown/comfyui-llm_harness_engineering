"""Import the pack with its production package layout and one shared cache."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK.parent), str(PACK.parents[1])]

from aiohttp import web
import server

with patch.object(server.PromptServer, "instance", SimpleNamespace(routes=web.RouteTableDef()), create=True):
    pack = importlib.import_module(PACK.name)

generation_cache = importlib.import_module(f"{PACK.name}.generation_cache")
reply = importlib.import_module(f"{PACK.name}.llm_remote_reply")
connections = importlib.import_module(f"{PACK.name}.image_api.connections")
request = importlib.import_module(f"{PACK.name}.image_api.request")
BatchCheckpoint = importlib.import_module(f"{PACK.name}.image_api.checkpoint").BatchCheckpoint
APIImageEdit = pack.NODE_CLASS_MAPPINGS["LLMHarnessImageEdit"]
APIImageGenerate = pack.NODE_CLASS_MAPPINGS["LLMHarnessImageGenerate"]
