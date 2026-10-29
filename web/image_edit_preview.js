import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// The optional local gallery shares zoom behavior with liuxinyu's image nodes.
// The pack still works independently with the native gallery when unavailable.
const smoothPreview = import("/extensions/comfyui-liuxinyu-packs/smooth_image_grid.js").catch(() => null);

// Only file descriptors live here; the gallery owns decoded preview images.
const previews = new Map();
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

function showProgress(node, progress) {
  if (!node || !progress) return;
  node.harnessImageProgress = progress;
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "LLMHarness.ImageEditProgressPreview",
  async nodeCreated(node) {
    if (node.comfyClass !== "LLMHarnessImageEdit") return;
    const widget = node.addCustomWidget({
      name: "llm_harness_total_progress", type: "llm_harness_total_progress", serialize: false, options: { serialize: false },
      computeSize: () => [0, 32],
      draw(ctx, owner, width, y) {
        const value = owner.harnessImageProgress;
        ctx.save();
        ctx.fillStyle = "#252d28";
        ctx.fillRect(10, y + 3, width - 20, 25);
        if (value?.total > 0) {
          ctx.fillStyle = "#376646";
          ctx.fillRect(10, y + 3, (width - 20) * Math.min(1, value.completed / value.total), 25);
        }
        ctx.fillStyle = "#eee";
        ctx.font = "13px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(value ? `已完成 ${value.completed} / ${value.total} 张` : "总进度 · 等待运行", width / 2, y + 20);
        ctx.restore();
      },
    });
    const index = node.widgets.indexOf(widget);
    node.widgets.splice(index, 1);
    node.widgets.splice(node.widgets.findIndex((item) => item.name === "force_regenerate") + 1, 0, widget);
    const executed = node.onExecuted;
    node.onExecuted = function (result) {
      const previous = executed?.apply(this, arguments);
      showProgress(this, result?.llm_harness_image_progress?.[0]);
      return previous;
    };
    const started = node.onExecutionStart;
    node.onExecutionStart = function (...args) {
      this.harnessImageProgress = null;
      return started?.apply(this, args);
    };
    let removed = false;
    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) { removed = true; return onRemoved?.apply(this, args); };
    const preview = await smoothPreview;
    if (preview && !removed) node.harnessImageGallery = preview.attachSmoothImagePreview(node, app, api);
  },
  setup() {
    api.addEventListener("execution_start", ({ detail }) => {
      previews.clear();
      promptId = detail.prompt_id;
      graphId = app.rootGraph.id;
    });
    api.addEventListener("llm_harness_image_preview", ({ detail }) => {
      if (detail.prompt_id !== promptId || app.rootGraph.id !== graphId) return;
      const nodeId = String(detail.node);
      let batches = previews.get(nodeId);
      if (!batches) {
        batches = new Map();
        previews.set(nodeId, batches);
      }
      batches.set(detail.batch, detail.images);
      const images = [...batches.entries()].sort(([a], [b]) => a - b)
        .flatMap(([, files]) => files);
      // Keep output history and incremental previews in sync without marking
      // the still-running node as executed.
      app.nodeOutputs[nodeId] = { images };
      const node = findNode(nodeId);
      node?.harnessImageGallery?.setImages(images);
      showProgress(node, { completed: detail.completed, total: detail.total });
      app.canvas.setDirty(true, true);
    });
    api.addEventListener("executed", ({ detail }) => {
      previews.delete(String(detail.display_node || detail.node));
    });
  },
  beforeConfigureGraph() {
    previews.clear();
    promptId = null;
    graphId = null;
  },
});
