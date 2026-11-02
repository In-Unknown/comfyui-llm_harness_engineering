"""LLM Harness Engineering - ComfyUI custom node pack."""

from .llm_remote_reply import LLMRemoteReply
from .boolean_return import NODE_CLASSES as RETURN_NODE_CLASSES
from .image_api import NODE_CLASSES as IMAGE_NODE_CLASSES

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node_class in (LLMRemoteReply, *RETURN_NODE_CLASSES, *IMAGE_NODE_CLASSES):
    schema = node_class.define_schema()
    NODE_CLASS_MAPPINGS[schema.node_id] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name

print(f"[comfyui-llm_harness-engineering] Loaded {len(NODE_CLASS_MAPPINGS)} nodes.")
