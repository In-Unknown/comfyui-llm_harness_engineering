import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LLMHarness.ImageSources",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["llm_remote_reply", "LLMHarnessImageEdit", "LLMHarnessImageGenerate"].includes(nodeData.name)) return;
        const isTextNode = nodeData.name === "llm_remote_reply";
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            const source = node.widgets.find(widget => widget.name === "connection");
            const model = node.widgets.find(widget => widget.name === "model");
            if (!source || !model) return result;
            let removed = false;
            let revision = 0;
            async function refresh() {
                const current = ++revision;
                try {
                    const response = await fetch("/llm_harness/connections", { cache: "no-store" });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const { connections } = await response.json();
                    if (removed || current !== revision) return;
                    const names = Object.keys(connections).sort();
                    source.options.values = names.length ? names : ["（请添加来源）"];
                    if (!names.includes(source.value)) {
                        if (source.value && !source.value.startsWith("（")) {
                            source.options.values = [source.value, ...names];
                            node.setDirtyCanvas(true, true);
                            return;
                        }
                        source.value = source.options.values[0];
                    }
                    const cfg = connections[source.value];
                    const models = cfg?.models?.length ? cfg.models : cfg?.model ? [cfg.model] : ["（未配置模型）"];
                    model.options.values = models;
                    if (!models.includes(model.value)) model.value = models[0];
                    node.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error("[LLM Harness] 来源刷新失败", error);
                }
            }
            const changed = source.callback;
            source.callback = function (...values) {
                changed?.apply(this, values);
                void refresh();
            };
            const editButton = node.addWidget("button", isTextNode ? "＋ 添加 / 编辑连接" : "添加 / 编辑来源", null, async () => {
                try {
                    const response = await fetch("/llm_harness/connections/add", { method: "POST" });
                    const data = await response.json();
                    if (!data.ok) throw new Error(data.error || "无法打开来源管理窗口");
                } catch (error) {
                    alert(`打开来源管理失败：${error.message}`);
                }
            }, { serialize: false });
            editButton.serialize = false;
            const refreshButton = node.addWidget("button", isTextNode ? "刷新连接" : "刷新来源", null, refresh, { serialize: false });
            refreshButton.serialize = false;
            const configure = node.onConfigure;
            node.onConfigure = function (...values) {
                const value = configure?.apply(this, values);
                void refresh();
                return value;
            };
            const onFocus = () => void refresh();
            window.addEventListener("focus", onFocus);
            const dispose = node.onRemoved;
            node.onRemoved = function (...values) {
                removed = true;
                window.removeEventListener("focus", onFocus);
                return dispose?.apply(this, values);
            };
            void refresh();
            return result;
        };
    },
});
