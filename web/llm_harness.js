import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function openDialog() {
    try {
        const r = await fetch("/llm_harness/connections/add", { method: "POST" });
        const j = await r.json();
        if (!j.ok) alert("打开窗口失败：" + (j.error || "未知错误"));
        return !!j.ok;
    } catch (e) {
        alert("请求失败：" + e);
        return false;
    }
}

app.registerExtension({
    name: "LLMHarness.Connections",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "llm_remote_reply") return;
        const origCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            const node = this;

            const connWidget = node.widgets.find(w => w.name === "connection");
            const modelWidget = node.widgets.find(w => w.name === "model");

            if (connWidget && modelWidget) {
                let cacheData = null;
                const updateModels = async () => {
                    try {
                        if (!cacheData) {
                            const res = await fetch("/llm_harness/connections");
                            const json = await res.json();
                            cacheData = json.connections;
                        }
                        const cfg = cacheData[connWidget.value];
                        let models = [];
                        if (cfg) {
                            models = cfg.models || [];
                            if (models.length === 0 && cfg.model) models = [cfg.model];
                        }
                        if (models.length === 0) models = ["（未配置模型）"];

                        modelWidget.options.values = models;
                        if (!models.includes(modelWidget.value)) {
                            modelWidget.value = models[0];
                        }
                        node.setSize(node.size);
                        app.graph.setDirtyCanvas(true, true);
                    } catch (e) {
                        console.error("[LLM Harness] failed to update models", e);
                    }
                };

                const origCb = connWidget.callback;
                connWidget.callback = function (value) {
                    if (origCb) origCb.apply(this, arguments);
                    updateModels();
                };
                updateModels();
            }

            node.addWidget("button", "＋ 添加 / 编辑连接", null, async () => {
                await openDialog();
            });

            // 完全复刻标准实现：外层 flex 容器，内层滚动区
            const panel = document.createElement("div");
            panel.style.cssText = "display: flex; flex-direction: column; overflow: hidden; box-sizing: border-box;";

            const body = document.createElement("div");
            body.style.cssText = "flex: 1; min-width: 0; min-height: 0; overflow-y: auto; background: #1e1e1e; color: #4ade80; font-family: monospace; font-size: 12px; border-radius: 4px; padding: 6px; box-sizing: border-box; white-space: pre-wrap; word-break: break-all; outline: none; user-select: text;";
            panel.appendChild(body);

            // 原生滚轮拦截，转发给画布
            panel.addEventListener("wheel", (event) => {
                event.preventDefault();
                event.stopPropagation();
                if (!event.ctrlKey) {
                    const unit = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? body.clientHeight : 1;
                    body.scrollTop += (event.deltaY || event.deltaX) * unit;
                } else {
                    app.canvas.canvas.dispatchEvent(new WheelEvent("wheel", {
                        deltaX: event.deltaX, deltaY: event.deltaY, deltaZ: event.deltaZ,
                        deltaMode: event.deltaMode, clientX: event.clientX, clientY: event.clientY,
                        ctrlKey: event.ctrlKey, altKey: event.altKey, metaKey: event.metaKey,
                        bubbles: true, cancelable: true,
                    }));
                }
            }, { passive: false });

            let widget;
            widget = node.addDOMWidget("stream_output", "流式预览", panel, {
                hideOnZoom: false,
                getMinHeight: () => node._isExpanded ? 160 : 0,
                onDraw: fitPanel,
                afterResize: () => {
                    if (widget) widget.width = undefined;
                    fitPanel();
                }
            });
            widget.serialize = false;

            function fitPanel() {
                const w = Math.max(0, node.size[0] - 20);
                panel.style.width = `${w}px`;
                panel.style.display = node._isExpanded ? "flex" : "none";
            }

            const origSelected = node.onSelected;
            node.onSelected = function() {
                if (origSelected) origSelected.apply(this, arguments);
                setTimeout(fitPanel, 5);
                setTimeout(fitPanel, 50);
            };

            const origDeselected = node.onDeselected;
            node.onDeselected = function() {
                if (origDeselected) origDeselected.apply(this, arguments);
                setTimeout(fitPanel, 5);
                setTimeout(fitPanel, 50);
            };

            setTimeout(fitPanel, 50);

            node._streamPanel = body;
            node._forceSize = fitPanel;
            return r;
        };
    },
    async setup() {
        api.addEventListener("llm_harness.stream", (event) => {
            const { node_id, text } = event.detail;
            const node = app.graph.getNodeById(node_id);
            if (node && node._streamPanel) {
                if (!node._isExpanded) {
                    node._isExpanded = true;
                    node._streamPanel.innerText = "";
                    node.setSize([node.size[0], node.computeSize()[1]]);
                    if (node._forceSize) node._forceSize();
                }
                node._streamPanel.innerText += text;
                node._streamPanel.scrollTop = node._streamPanel.scrollHeight;
            }
        });

        api.addEventListener("execution_start", () => {
            app.graph._nodes.forEach(n => {
                if (n.type === "llm_remote_reply" && n._streamPanel) {
                    n._isExpanded = false;
                    n._streamPanel.innerText = "";
                    n.setSize([n.size[0], n.computeSize()[1]]);
                    if (n._forceSize) n._forceSize();
                }
            });
        });
    }
});
