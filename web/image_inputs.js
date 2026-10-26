import { app } from "../../scripts/app.js";

let hasDynamicImages = false;

// 改输入名时保留原槽位顺序和 link，旧画布无需重新接三条图片线。
export function migrateImageInputs(graphData) {
    for (const node of graphData.nodes ?? []) {
        if (node.type !== "llm_remote_reply") continue;
        node.inputs ??= [];
        for (const input of node.inputs) {
            const match = /^image_([123])$/.exec(input.name);
            if (match) {
                input.name = `images.图片${match[1]}`;
                input.label = `图片${match[1]}`;
            }
        }
        if (!node.inputs.some(input => input.name === "image_list")) {
            node.inputs.push({name: "image_list", type: "IMAGE", label: "列表合并图片", link: null});
        }
    }
    for (const subgraph of graphData.definitions?.subgraphs ?? []) {
        migrateImageInputs(subgraph);
    }
}

app.registerExtension({
    name: "LLMHarness.ImageInputs",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "llm_remote_reply") {
            hasDynamicImages = nodeData.input?.required?.images?.[0] === "COMFY_AUTOGROW_V3";
        }
    },
    beforeConfigureGraph(graphData) {
        if (hasDynamicImages) migrateImageInputs(graphData);
    },
});
