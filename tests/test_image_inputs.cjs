const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let extension;
const source = fs.readFileSync(path.join(__dirname, "../web/image_inputs.js"), "utf8")
    .replace(/^import .*;$/m, "").replace("export function", "function");
vm.runInNewContext(source, {app: {registerExtension(value) { extension = value; }}});

const oldNode = {type: "llm_remote_reply", inputs: [
    {name: "prompt", type: "STRING", link: 0},
    {name: "image_1", type: "IMAGE", link: 10},
    {name: "image_2", type: "IMAGE", link: null},
    {name: "image_3", type: "IMAGE", link: 12},
]};
const graph = {nodes: [oldNode, {type: "another_node", inputs: [{name: "image_1", link: 20}]}],
    definitions: {subgraphs: [{nodes: [{type: "llm_remote_reply", inputs: [{name: "image_2", link: 21}]}]}]}};
extension.beforeConfigureGraph(graph);
assert.equal(oldNode.inputs[1].name, "image_1", "Do not migrate while the server is still running the old schema");
extension.beforeRegisterNodeDef(null, {name: "llm_remote_reply", input: {required: {images: ["COMFY_AUTOGROW_V3"]}}});
extension.beforeConfigureGraph(graph);
assert.deepEqual(oldNode.inputs.map(input => input.name),
    ["prompt", "images.图片1", "images.图片2", "images.图片3", "image_list"]);
assert.deepEqual(oldNode.inputs.map(input => input.link), [0, 10, null, 12, null]);
assert.equal(graph.nodes[1].inputs[0].name, "image_1");
assert.equal(graph.definitions.subgraphs[0].nodes[0].inputs[0].name, "images.图片2");
const migrated = JSON.stringify(graph);
extension.beforeConfigureGraph(graph);
assert.equal(JSON.stringify(graph), migrated);
console.log("Image input migration: links, order, subgraphs and repeat loading passed.");
