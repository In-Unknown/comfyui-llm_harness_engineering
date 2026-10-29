import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let promptId = null;
let graphId = null;

function findNode(executionId) {
    let graph = app.rootGraph;
    const path = String(executionId).split(":");
    for (let index = 0; index < path.length; index++) {
        const node = graph?.getNodeById(path[index]);
        if (index === path.length - 1) return node;
        graph = node?.subgraph;
    }
}

function clearPreviews(graph) {
    for (const node of graph?._nodes ?? []) {
        if (node.type === "llm_remote_reply" && node._streamPanel) {
            node._isExpanded = false;
            node._streamPanel.innerText = "";
            node.setSize([node.size[0], node.computeSize()[1]]);
            node._forceSize?.();
        }
        if (node.subgraph) clearPreviews(node.subgraph);
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
                serialize: false,
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
            if (event.detail.prompt_id !== promptId || app.rootGraph.id !== graphId) return;
            const { node_id, text } = event.detail;
            const node = findNode(node_id);
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

        api.addEventListener("execution_start", ({ detail }) => {
            promptId = detail.prompt_id;
            graphId = app.rootGraph.id;
            clearPreviews(app.rootGraph);
        });
    },
    beforeConfigureGraph() {
        promptId = null;
        graphId = null;
    }
});
