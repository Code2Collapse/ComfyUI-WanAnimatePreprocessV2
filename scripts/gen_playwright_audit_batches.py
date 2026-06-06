import json
from pathlib import Path

manifest = json.loads(
    (Path(__file__).parent / "_playwright_node_manifest.json").read_text(encoding="utf-8")
)
nodes = manifest["loaded"]
out_dir = Path(__file__).resolve().parents[2] / ".playwright-mcp"
out_dir.mkdir(parents=True, exist_ok=True)
chunk = 24

template = '''async (page) => {{
  const nodes = {nodes_json};
  const ready = await page.evaluate(() => !!(window.app?.graph && window.LiteGraph?.createNode));
  if (!ready) return {{ error: "comfy_not_ready", batch: {batch_id} }};

  const results = [];
  for (const className of nodes) {{
    const r = await page.evaluate(async (className) => {{
      const lg = window.LiteGraph;
      const app = window.app;
      const vue = !!lg.vueNodesMode;
      if (!lg?.createNode || !app?.graph) return {{ className, ok: false, error: "no_app" }};
      if (!lg.registered_node_types?.[className]) return {{ className, ok: false, error: "not_registered" }};
      let node;
      try {{
        node = lg.createNode(className);
        app.graph.add(node, false);
        const n = (app.graph._nodes || []).length;
        node.pos = [40 + (n % 4) * 380, 40 + Math.floor(n / 4) * 220];
        const sz = node.computeSize?.() || [300, 180];
        node.setSize?.(sz);
        if (vue && app.canvas?.selectNode) app.canvas.selectNode(node);
      }} catch (e) {{
        return {{ className, ok: false, error: String(e?.message || e) }};
      }}
      await new Promise((r) => setTimeout(r, vue ? 220 : 120));
      const errors = [];
      const h = node.size?.[1] || 0;
      const w = node.size?.[0] || 0;
      if (h < 24 || w < 80) errors.push("bad_compute_size");
      const el = node.el;
      const rect = el?.getBoundingClientRect?.();
      if (h > 1600) errors.push("oversized");
      if (className === "WanFaceController3DV2") {{
        const ed = document.querySelector(".fc3d-editor-root");
        if (!ed) errors.push("no_fc3d_editor_dom");
        const last = node.widgets?.[node.widgets.length - 1]?.name;
        if (last !== "face_overlay") errors.push("dom_not_last:" + (last || "?"));
        const vis = (node.widgets || []).filter((wg) => {{
          if (wg.name === "face_overlay" || wg.__fc3d_chrome_hidden) return false;
          return wg.element && wg.element.style?.display !== "none" && !wg.hidden;
        }}).length;
        if (vis > 2) errors.push("chrome_visible:" + vis);
        if (ed && rect) {{
          const edR = ed.getBoundingClientRect();
          if (rect.height > 700 && edR.height < rect.height * 0.55) errors.push("black_slab");
        }} else if (ed && h > 900) {{
          const edH = ed.getBoundingClientRect?.()?.height || 0;
          if (edH > 0 && edH < h * 0.55) errors.push("black_slab_vs_size");
        }}
      }}
      return {{
        className,
        ok: !errors.length,
        errors,
        vue,
        size: node.size ? [...node.size] : null,
        rect: rect ? {{ w: Math.round(rect.width), h: Math.round(rect.height) }} : null,
      }};
    }}, className);
    results.push(r);
  }}
  return {{ batch: {batch_id}, vue: true, tested: results.length, passed: results.filter((x) => x.ok).length, failed: results.filter((x) => !x.ok) }};
}}
'''

for i in range(0, len(nodes), chunk):
    part = nodes[i : i + chunk]
    bid = i // chunk
    (out_dir / f"comfy_audit_b{bid}.js").write_text(
        template.format(nodes_json=json.dumps(part), batch_id=bid),
        encoding="utf-8",
    )

# Face controller interaction script
(out_dir / "comfy_face_interact.js").write_text(
    '''async (page) => {
  const r = await page.evaluate(async () => {
    const app = window.app;
    const lg = window.LiteGraph;
    if (!app?.graph) return { ok: false, error: "no_graph" };
    let node = (app.graph._nodes || []).find((n) => n.comfyClass === "WanFaceController3DV2");
    if (!node) {
      node = lg.createNode("WanFaceController3DV2");
      app.graph.add(node, false);
      node.pos = [200, 200];
      node.setSize?.(node.computeSize?.() || [380, 480]);
    }
    app.canvas?.selectNode?.(node);
    await new Promise((r) => setTimeout(r, 600));
    const root = document.querySelector(".fc3d-editor-root");
    if (!root) return { ok: false, error: "no_editor_root", size: node.size };
    const tabs = {};
    for (const b of root.querySelectorAll("button")) {
      const t = b.textContent?.trim();
      if (["Face","Expr","Gaze","Pose","Settings"].includes(t)) tabs[t] = true;
    }
    const happy = Array.from(root.querySelectorAll("button")).find((b) => b.textContent === "Happy");
  if (happy) happy.click();
    await new Promise((r) => setTimeout(r, 400));
    const canvas = root.querySelector("canvas");
    const edR = root.getBoundingClientRect();
    const cR = canvas?.getBoundingClientRect?.();
    return {
      ok: true,
      vue: !!lg.vueNodesMode,
      size: node.size ? [...node.size] : null,
      tabs: Object.keys(tabs),
      editor: { w: Math.round(edR.width), h: Math.round(edR.height) },
      canvas: cR ? { w: Math.round(cR.width), h: Math.round(cR.height) } : null,
      lastWidget: node.widgets?.[node.widgets.length - 1]?.name,
      hiddenChrome: (node.widgets || []).filter((w) => w.__fc3d_chrome_hidden).length,
    };
  });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2_live.png" });
  return r;
}
''',
    encoding="utf-8",
)
print("batches", (len(nodes) + chunk - 1) // chunk)
