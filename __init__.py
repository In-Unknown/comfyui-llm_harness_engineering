"""LLM Harness Engineering - ComfyUI custom node pack."""

import importlib.util
import os
import sys

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = os.path.dirname(__file__)

for filename in sorted(os.listdir(current_dir)):
    if not filename.endswith(".py") or filename.startswith("__"):
        continue

    module_name = filename[:-3]
    full_module_name = f"_comfyui_llm_harness_{module_name}"

    if full_module_name in sys.modules:
        continue

    file_path = os.path.join(current_dir, filename)
    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None or spec.loader is None:
        continue

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[comfyui-llm_harness-engineering] Error loading {filename}: {e}")
        continue

    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if not isinstance(obj, type):
            continue

        if hasattr(obj, "define_schema") and hasattr(obj, "execute"):
            try:
                schema = obj.define_schema()
                node_id = getattr(schema, "node_id", None)
                if node_id:
                    display_name = getattr(schema, "display_name", node_id)
                    NODE_CLASS_MAPPINGS[node_id] = obj
                    NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name
            except Exception:
                pass

        elif hasattr(obj, "INPUT_TYPES") and hasattr(obj, "FUNCTION"):
            node_id = obj.__name__
            display_name = getattr(obj, "DISPLAY_NAME", node_id)
            NODE_CLASS_MAPPINGS[node_id] = obj
            NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name

from .image_api import NODE_CLASSES as IMAGE_NODE_CLASSES

for node_class in IMAGE_NODE_CLASSES:
    schema = node_class.define_schema()
    NODE_CLASS_MAPPINGS[schema.node_id] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name

print(f"[comfyui-llm_harness-engineering] Loaded {len(NODE_CLASS_MAPPINGS)} nodes.")
