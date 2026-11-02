"""Re-run a reply's validation graph with conversational error feedback."""

from dataclasses import replace

from comfy_api.latest import io
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy_execution.graph_utils import GraphBuilder, is_link

from . import llm_remote_reply as reply


def _validation_graph(prompt, source_id, end_id):
    ancestors, children = {}, {}
    pending = [end_id]
    while pending:
        key = pending.pop()
        if key == source_id or key in ancestors:
            continue
        node = prompt.get_node(key)
        ancestors[key] = node
        for value in node.get("inputs", {}).values():
            if is_link(value):
                children.setdefault(value[0], set()).add(key)
                pending.append(value[0])

    body = {}
    pending = list(children.get(source_id, ()))
    while pending:
        key = pending.pop()
        if key in body:
            continue
        node = ancestors[key]
        if "_loop_end" in node:
            raise ValueError("回复校验子流中暂不支持嵌套循环边界，请在循环内部完成一次回复校验。")
        body[key] = node
        pending.extend(children.get(key, ()))
    return body


class LLMHarnessBooleanReturn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMHarnessBooleanReturn", display_name="布尔条件返回", category="llm_harness",
            description=("调度口直接连接 LLM 远程回复。校验通过时输出最终回复及已用重试次数；"
                         "失败时携带原始提示词、系统提示词、图片、历次回复和错误信息重新请求，"
                         "并重新执行校验子流。最大重试次数不包含首次请求，0 表示失败立即报错。"
                         "校验节点须返回 false 而非抛异常；后续处理接本节点的回复输出。"
                         "每轮只接受单条回复和一个布尔值，多项执行列表请先分包。"),
            is_output_node=True, enable_expand=True, is_input_list=True,
            inputs=[
                reply.SCHEDULE.Input("schedule", display_name="调度输入"),
                io.Boolean.Input("passed", display_name="校验通过", force_input=True,
                                 tooltip="true 放行；false 重试或报错。应由本轮回复的校验子流产生。"),
                io.Int.Input("max_retries", display_name="最大重试次数", default=3, min=0,
                             tooltip="首次回复之外允许的追加请求数，必须为非负整数；0 表示校验失败立即停止。"),
                io.String.Input("error_message", display_name="报错信息", default="回复未满足要求，请检查格式或数值范围。",
                                multiline=True, tooltip="可填写文本或转换为输入接入本轮校验的错误信息；重试时在原文前后自动添加失败说明与完整重写要求，组成 user 反馈消息。全空白使用默认错误。详见节点帮助。"),
            ],
            outputs=[io.String.Output("text", display_name="回复"),
                     io.Int.Output("retry_count", display_name="重试次数")],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, schedule, passed, max_retries, error_message):
        throw_exception_if_processing_interrupted()
        inputs = dict(schedule=schedule, passed=passed, max_retries=max_retries, error_message=error_message)
        if any(len(values) != 1 for values in inputs.values()):
            raise ValueError("布尔条件返回每轮必须接收一条回复、一个布尔值、一个重试上限和一条错误信息；空或多项执行列表请先整理或分包。")
        schedule, passed, max_retries, error_message = (values[0] for values in inputs.values())
        if not isinstance(schedule, reply.ReplySchedule):
            raise ValueError("调度输入必须直接连接 LLM 远程回复的调度输出。")
        if type(passed) is not bool:
            raise ValueError("校验通过必须是 BOOLEAN 布尔值，不能用字符串或数值代替。")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("最大重试次数必须是非负整数，0 表示失败立即报错。")
        if not isinstance(error_message, str):
            raise ValueError("报错信息必须是 STRING 字符串。")
        prompt, end_id = cls.hidden.dynprompt, cls.hidden.unique_id
        if prompt.get_node(end_id)["inputs"].get("schedule") != [schedule.source_id, 1]:
            raise ValueError("调度输入必须直接连接对应 LLM 远程回复，不支持中间处理或交叉配对。")
        retry_count = len(schedule.history)
        if passed:
            return io.NodeOutput(schedule.text, retry_count)
        if not error_message.strip():
            error_message = "回复未通过校验。"
        if retry_count >= max_retries:
            raise RuntimeError(f"LLM 回复校验失败：已重试 {retry_count}/{max_retries} 次。{error_message}")

        body = _validation_graph(prompt, schedule.source_id, end_id)
        graph = GraphBuilder()
        state = replace(schedule, history=(*schedule.history, (schedule.text, error_message)))
        source = graph.node("LLMHarnessReplyRetry", "reply", state=state)
        source.set_override_display_id(prompt.get_display_node_id(schedule.source_id))
        copies = {key: graph.node(node["class_type"], f"check_{index}")
                  for index, (key, node) in enumerate(body.items())}

        def copied_link(value):
            if is_link(value):
                if value[0] == schedule.source_id:
                    return source.out(value[1])
                if value[0] in copies:
                    return copies[value[0]].out(value[1])
            return value

        for key, copy in copies.items():
            copy.set_override_display_id(prompt.get_display_node_id(key))
            copy.inputs.update({name: copied_link(value) for name, value in body[key]["inputs"].items()})
        # Keep the original budget even if its input was computed from the reply.
        copies[end_id].set_input("max_retries", max_retries)
        return io.NodeOutput(copies[end_id].out(0), copies[end_id].out(1), expand=graph.finalize())


class LLMHarnessReplyRetry(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMHarnessReplyRetry", is_dev_only=True,
            inputs=[io.AnyType.Input("state")],
            outputs=[io.String.Output(), reply.SCHEDULE.Output()],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, state):
        throw_exception_if_processing_interrupted()
        cfg = reply._load_connections().get(state.connection)
        if not cfg:
            raise ValueError(f"未找到连接「{state.connection}」，请恢复连接配置后重新运行。")
        node_id = cls.hidden.dynprompt.get_display_node_id(cls.hidden.unique_id)
        text = reply._call_llm(cfg, **state.request, images_b64=state.images_b64,
                              node_id=node_id, history=state.history)
        throw_exception_if_processing_interrupted()
        return io.NodeOutput(text, replace(state, source_id=cls.hidden.unique_id, text=text))


NODE_CLASSES = (LLMHarnessBooleanReturn, LLMHarnessReplyRetry)
