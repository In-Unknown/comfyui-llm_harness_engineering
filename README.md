# ComfyUI LLM Harness Engineering

在 ComfyUI 中调用远程文字、视觉和图像 API，自行管理来源、模型和 Key。

## 节点

| 节点 | 功能 |
| --- | --- |
| LLM 远程回复 | 支持 Chat Completions、Responses、Anthropic Messages，流式显示回复；支持多个图片输入和执行列表合并输入。 |
| API图片编辑 | 可变数量图片输入、主图批次编辑、参考图、蒙版、并发请求、进度预览及断点续作。 |
| API图像生成 | 选择来源和模型，根据提示词生成图片。 |

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
- [API图片编辑：参考图、蒙版、并发和断点续作](web/docs/LLMHarnessImageEdit.md)
- [API图像生成：来源、参数和输出](web/docs/LLMHarnessImageGenerate.md)

图片节点说明也可以从 ComfyUI 节点属性面板的「信息」中查看。实际支持的模型、尺寸、图片数量和计费规则由所选服务决定。测试使用模拟接口，不代表所有服务商均已验证。

## 本地数据与版本控制

| 数据 | 保存位置 | Git 策略 |
| --- | --- | --- |
| 来源、模型和 API Key | 本包的 `data/connections.json` | 整个 `data/` 忽略，不提供真实配置样例。 |
| 编辑断点、响应和生成图片备份 | `ComfyUI/user/llm_harness_image_resume` | 位于仓库之外；同时忽略本包内同名运行目录及 `*.response.json`。 |
| 临时预览图 | ComfyUI 的 temp 目录 | 位于仓库之外；同时忽略本包内 `temp/` 等运行目录。 |
| Python 缓存、环境、日志 | 对应缓存或环境目录 | 由 `.gitignore` 排除。 |

Key 在本机配置文件中保存为明文，不写入节点参数；请妥善保管该文件。`.gitignore` 不会自动处理已跟踪文件，提交前可用 `git status --short` 和 `git diff --cached` 检查内容。不要使用 `git add -f` 强行加入配置、请求响应或生成文件。

## 开发检查

`tests/test_image_api.py` 使用 ComfyUI 环境测试图片来源切换、请求内容、蒙版和缓存隔离，网络请求均被模拟。`tests/test_image_inputs.py` 还需要旁边安装 `comfyui-liuxinyu-packs`，用于验证实际切块节点输出的执行列表。

前端行为测试使用 Node.js 执行 `tests/test_image_sources.cjs` 和 `tests/test_image_inputs.cjs`。这些测试不会产生付费请求。
