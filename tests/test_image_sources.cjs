const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let extension;
let pending = [];
const listeners = new Set();
const code = fs.readFileSync(path.join(__dirname, "../web/image_sources.js"), "utf8").replace(/^import .*;$/m, "");
vm.runInNewContext(code, {
    app: {registerExtension(value) { extension = value; }},
    fetch() { return new Promise(resolve => pending.push(resolve)); },
    window: {addEventListener(_, callback) { listeners.add(callback); }, removeEventListener(_, callback) { listeners.delete(callback); }},
    console, alert() { throw new Error("Unexpected dialog"); },
});
const tick = () => new Promise(resolve => setImmediate(resolve));
const connections = {A: {models: ["a1", "a2"]}, B: {models: ["b1"]}};
const reply = (resolve, data = connections) => resolve({ok: true, json: async () => ({connections: data})});

(async () => {
    function Node() {
        this.widgets = [{name: "connection", value: "A", options: {}}, {name: "model", value: "a2", options: {}}];
    }
    Node.prototype.addWidget = function (type, name, value, callback) { this.widgets.push({type, name, value, callback}); };
    Node.prototype.setDirtyCanvas = () => {};
    extension.beforeRegisterNodeDef(Node, {name: "LLMHarnessImageEdit"});
    const node = new Node();
    node.onNodeCreated();
    reply(pending.shift());
    await tick();
    const [source, model] = node.widgets;
    assert.equal(model.value, "a2", "Keep saved model when it belongs to the source");
    source.value = "B";
    source.callback();
    reply(pending.shift());
    await tick();
    assert.equal(model.value, "b1", "Changing source chooses a valid model");
    node.onConfigure();
    node.onConfigure();
    const [old, newest] = pending.splice(0);
    reply(newest, {B: {models: ["b2"]}});
    await tick();
    reply(old);
    await tick();
    assert.equal(model.value, "b2", "Stale requests cannot overwrite current source data");
    node.onConfigure();
    node.onRemoved();
    reply(pending.shift());
    await tick();
    assert.equal(model.value, "b2");
    assert.equal(listeners.size, 0, "Removal cleans up focus listener");
    function Other() {}
    extension.beforeRegisterNodeDef(Other, {name: "MXWQ_GPTImageEdit"});
    assert.equal(Other.prototype.onNodeCreated, undefined, "Original nodes are not modified");
    console.log("Image sources: saved model, switching, stale refresh, disposal and original isolation passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
