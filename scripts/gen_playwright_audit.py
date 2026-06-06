import json
from pathlib import Path

manifest = json.loads(
    (Path(__file__).parent / "_playwright_node_manifest.json").read_text(encoding="utf-8")
)
nodes = manifest["loaded"]
out = Path(__file__).resolve().parents[2] / ".playwright-mcp" / "comfy_audit_all.js"

body = f"""async (page) => {{
  const nodes = {json.dumps(nodes)};

  await page.waitForFunction(() => window.app?.graph && window.LiteGraph?.createNode, {{ timeout: 120000 }});

  const results = [];
  for (const className of nodes) {{
    const r = await page.evaluate(async (className) => {{
      const lg = window.LiteGraph;
      const app = window.app;
      if (!lg?.createNode || !app?.graph) return {{ className, ok: false, error: "no_app" }};
      if (!lg.registered_node_types?.[className]) return {{ className, ok: false, error: "not_registered" }};
      let node;
      try {{
        node = lg.createNode(className);
        app.graph.add(node, false);
        const n = (app.graph._nodes || []).length;
        node.pos = [40 + (n % 4) * 400, 40 + Math.floor(n / 4) * 240];
        node.setSize?.(node.computeSize?.() || [300, 180]);
        app.graph.setDirtyCanvas?.(true, true);
      }} catch (e) {{
        return {{ className, ok: false, error: String(e?.message || e) }};
      }}
      await new Promise((r) => setTimeout(r, 180));
      const el = node.el;
      const rect = el?.getBoundingClientRect?.();
      const errors = [];
      if (!rect || rect.height < 6) errors.push("zero_height");
      if (rect && rect.height > 1400) errors.push("oversized");
      if (className === "WanFaceController3DV2") {{
        const ed = el?.querySelector?.(".fc3d-editor-root");
        const edR = ed?.getBoundingClientRect?.();
        if (!ed) errors.push("no_editor");
        if (rect && edR && rect.height > 700 && edR.height < rect.height * 0.55) errors.push("black_slab");
        const vis = (node.widgets || []).filter((w) => {{
          if (w.name === "face_overlay" || w.__fc3d_chrome_hidden) return false;
          return w.element && w.element.style.display !== "none" && !w.hidden;
        }}).length;
        if (vis > 2) errors.push("chrome_visible:" + vis);
        const last = node.widgets?.[node.widgets.length - 1]?.name;
        if (last !== "face_overlay") errors.push("dom_not_last:" + last);
      }}
      return {{
        className,
        ok: !errors.length,
        errors,
        size: node.size ? [...node.size] : null,
        rect: rect ? {{ w: Math.round(rect.width), h: Math.round(rect.height) }} : null,
      }};
    }}, className);
    results.push(r);
  }}

  const fcNode = await page.evaluate(() => {{
    const app = window.app;
    const node = (app.graph._nodes || []).find((n) => n.comfyClass === "WanFaceController3DV2");
    if (node) app.canvas.selectNode(node);
    return !!node;
  }});
  if (fcNode) {{
    await page.waitForTimeout(400);
    await page.screenshot({{ path: "d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2.png" }});
    await page.evaluate(() => {{
      for (const b of document.querySelectorAll(".fc3d-editor-root button")) {{
        if (b.textContent === "Expr") {{ b.click(); break; }}
      }}
    }});
    await page.waitForTimeout(350);
    await page.screenshot({{ path: "d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2_expr.png" }});
    await page.evaluate(() => {{
      for (const b of document.querySelectorAll(".fc3d-editor-root button")) {{
        if (b.textContent === "Happy") {{ b.click(); break; }}
      }}
    }});
    await page.waitForTimeout(300);
    await page.screenshot({{ path: "d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2_happy.png" }});
  }}

  return {{
    tested: results.length,
    passed: results.filter((x) => x.ok).length,
    failed: results.filter((x) => !x.ok),
  }};
}}
"""
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(body, encoding="utf-8")
print("wrote", out, "nodes", len(nodes))
