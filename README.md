# ComfyUI LLM Harness Engineering

在 ComfyUI 中调用远程文字、视觉和图像 API，自行管理来源、模型和 Key。

## 节点

提供 4 个可添加节点；另注册 1 个仅供校验重试展开使用的内部节点，不列入下表。

| 节点 | 节点 ID | 分类 | 功能 |
| --- | --- | --- | --- |
| [LLM 远程回复](web/docs/llm_remote_reply.md) | `llm_remote_reply` | `llm_harness` | 三种文字接口、流式回复、多图及列表合并；调度口连接布尔条件返回。 |
| [布尔条件返回](web/docs/LLMHarnessBooleanReturn.md) | `LLMHarnessBooleanReturn` | `llm_harness` | 校验失败时携带原始上下文、模型回复和错误信息重试；通过才输出最终回复，次数耗尽报错；上限可为 0。 |
| [API图片编辑](web/docs/LLMHarnessImageEdit.md) | `LLMHarnessImageEdit` | `llm_harness/image` | 可变数量图片输入、主图批次编辑、参考图、蒙版、并发请求、进度预览及断点续作。 |
| [API图像生成](web/docs/LLMHarnessImageGenerate.md) | `LLMHarnessImageGenerate` | `llm_harness/image` | 选择来源和模型，根据提示词生成图片。 |

## 安装

需要支持 V3 节点及 Autogrow 输入的较新版本 ComfyUI。将本仓库克隆到 `ComfyUI/custom_nodes`，目录名保持 `comfyui-llm_harness_engineering`，使用 **ComfyUI 实际使用的 Python 环境**安装依赖，然后重启 ComfyUI 并刷新浏览器。

Windows 便携版可在便携版根目录执行：

```powershell
git clone https://github.com/In-Unknown/comfyui-llm_harness_engineering.git ComfyUI/custom_nodes/comfyui-llm_harness_engineering
.\python_embeded\python.exe -m pip install -r ComfyUI/custom_nodes/comfyui-llm_harness_engineering/requirements.txt
```

PyTorch、NumPy、Pillow 等由 ComfyUI 环境提供。PySide6 用于来源配置窗口，该窗口在 **运行 ComfyUI 的电脑**上打开，需要桌面环境。

## 配置来源

1. 添加节点，点击 **添加 / 编辑来源**（文字节点中为 **添加 / 编辑连接**）。
2. 填写来源名称、API 基础地址、Key 和模型 ID；多个模型以英文逗号分隔。
3. 保存后，图片节点点击 **刷新来源**，文字节点刷新浏览器，再选择来源和模型。

基础地址示例为 `https://example.com/v1`。图片节点自行追加 `/images/generations` 或 `/images/edits`。来源配置的 API 格式只影响文字节点。

**图片编辑兼容范围**：沿用原 mxwq 节点的 JSON 协议，以 `images: [{"image_url": "data:image/png;base64,..."}]` 发送图片。仅支持 multipart 文件上传的编辑接口不能直接使用。图像响应支持 `data[0].b64_json` 或 HTTPS 的 `data[0].url`。

图片节点是原 mxwq 节点的独立副本，不需要安装原包，也不会修改原包的节点或配置。如果同时安装 `comfyui-liuxinyu-packs`，编辑节点可使用其流畅图片预览；未安装时使用普通预览。

## 使用说明

- [LLM 图片输入：批次、执行列表与合并](docs/ImageInputs.md)
- [LLM 远程回复：配置、输入输出和接口格式](web/docs/llm_remote_reply.md)
- [布尔条件返回：JSON 校验、错误反馈和重试次数](web/docs/LLMHarnessBooleanReturn.md)
- [API图片编辑：参考图、蒙版、并发和断点续作](web/docs/LLMHarnessImageEdit.md)
- [API图像生成：来源、参数和输出](web/docs/LLMHarnessImageGenerate.md)

节点说明也可以从 ComfyUI 节点属性面板的「信息」中查看。实际支持的模型、尺寸、图片数量和计费规则由所选服务决定。测试使用模拟接口，不代表所有服务商均已验证。

### 回复校验与反馈重试

LLM 的 `调度` 直接接布尔条件返回，`回复文本` 接校验子流；校验的 BOOLEAN 和错误 STRING 接返回节点。false 时追加原始上下文、历次回复与错误，重新请求并重跑校验；true 时输出最终回复及已用重试次数。上限不含首次请求，0 表示首次失败立即报错，不能为负数。

后续处理接**布尔条件返回的回复输出**；需要结构化数据时在其后重新解析。校验子流必须返回 false 而不是直接抛异常。每轮支持一条回复，多项执行列表请先分包；图片批次和列表合并图片仍可随单次请求携带。

### 全部生成式节点共用会话缓存

**LLM 远程回复、API图像生成、API图片编辑** 均默认启用。全部有效输入参与 SHA-256：来源名称、地址、Key、模型、文本和节点参数，以及各图片端口、蒙版与列表的原始内容、尺寸、数据类型和顺序。文字节点还包含 API 格式。重新生成开关与隐藏的节点 ID 不计入哈希；不同节点类型分别缓存。

每组成功结果保留到 ComfyUI 进程退出；输入 A→B→A、同类节点复制、循环和执行列表都可复用。同一组输入并发执行时只发出一次请求，不同输入仍可并行。图片编辑按执行项的完整输入及主图序号保存每张成功图片，失败后只补做缺失项。

缓存保存在内存，文字保存回复字符串，图片保存无损压缩 PNG，不长期保留输入图片或输出张量、不占显存；内存占用随不同结果数量增长。命中时跳过远程请求，重新显示回复或图片预览。文字节点的 **强制重新请求** 和图片节点的 **重新生成** 会绕过已有结果；成功后更新对应缓存，失败或中断不会覆盖对应的成功结果。

布尔校验的修正请求是例外：始终重新调用服务，不读取或覆盖首次回复缓存。再次排队会从初始回复和重试计数 0 开始，初始回复仍可命中缓存；不同校验规则不会相互覆盖原始回复。详见 [缓存中的反馈重试约定](docs/GenerationCache.md#布尔校验反馈重试)。

**重启 ComfyUI 后所有节点的自动复用记录都会清空。** 图片编辑的磁盘备份仍保留，但只会自动续做本次启动中的任务，不读取以前启动时的备份或旧版持久缓存。更新代码后需重启 ComfyUI 并刷新页面。

## 本地数据与版本控制

| 数据 | 保存位置 | Git 策略 |
| --- | --- | --- |
| 来源、模型和 API Key | 本包的 `data/connections.json` | 整个 `data/` 忽略，不提供真实配置样例。 |
| 生成输入哈希与文字／图片缓存 | 仅保存在 ComfyUI 进程内存，重启清空 | 不产生缓存文件。 |
| 编辑断点、响应和生成图片备份 | `ComfyUI/user/llm_harness_image_resume` | 位于仓库之外；同时忽略本包内同名运行目录及 `*.response.json`。 |
| 临时预览图 | ComfyUI 的 temp 目录 | 位于仓库之外；同时忽略本包内 `temp/` 等运行目录。 |
| Python 缓存、环境、日志 | 对应缓存或环境目录 | 由 `.gitignore` 排除。 |

Key 在本机配置文件中保存为明文，不写入节点参数；请妥善保管该文件。`.gitignore` 不会自动处理已跟踪文件，提交前可用 `git status --short` 和 `git diff --cached` 检查内容。不要使用 `git add -f` 强行加入配置、请求响应或生成文件。

## 开发检查

生成式节点共用 [generation_cache.py](generation_cache.py)，新增节点的接入约定见 [生成缓存基础](docs/GenerationCache.md)。`tests/test_generation_cache.py` 验证并发去重、完整输入匹配、强制更新、实际执行器及新进程隔离。

`tests/test_image_api.py` 使用 ComfyUI 环境测试图片来源切换、请求内容、蒙版和缓存隔离，网络请求均被模拟。`tests/test_image_inputs.py` 还需要旁边安装 `comfyui-liuxinyu-packs`，用于验证实际切块节点输出的执行列表。

`tests/test_boolean_return.py` 使用实际执行器验证三种接口的反馈历史、图片保留、校验子图重跑、重试上限、取消、缓存及下游等待；文档示例的集成测试使用旁边包中的 JSON 记录校验节点。`LLMHarnessReplyRetry` 仅供运行时展开使用，按本包入口注册并标记为内部节点，不列为用户可添加节点。

前端行为测试使用 Node.js 执行 `tests/test_image_sources.cjs` 和 `tests/test_image_inputs.cjs`。这些测试不会产生付费请求。

Windows 便携版可在便携版根目录执行以下完整检查。后端使用 ComfyUI 自带解释器，前端需要 Node.js；相邻的 `comfyui-liuxinyu-packs` 仅用于集成测试，运行本包节点不要求安装它。

```powershell
.\python_embeded\python.exe -B -s -X utf8 -m unittest discover -s ComfyUI/custom_nodes/comfyui-llm_harness_engineering/tests -v
node --test ComfyUI/custom_nodes/comfyui-llm_harness_engineering/tests/test_image_inputs.cjs ComfyUI/custom_nodes/comfyui-llm_harness_engineering/tests/test_image_sources.cjs
```

提交前可以清理 `__pycache__/` 与 `*.pyc`，它们会自动重新生成。保留 `tests/` 中的测试源码及测试辅助模块；本地 `data/` 是连接配置，不作为垃圾删除或提交。上述检查验证模拟服务下的行为，不代表已测试真实服务、来源配置窗口或浏览器交互。
