// Playwright MCP batch runner: async (page) => { ... }
// Usage: browser_run_code_unsafe with filename + edit OFFSET/LIMIT below.

const OFFSET = 0;
const LIMIT = 96;

async (page) => {
  const fs = await import("fs");
  const path = await import("path");
  const manifestPath = path.join(
    "D:/PROJECT/Custom_Nodes/ComfyUI-WanAnimatePreprocessV2/scripts",
    "_playwright_node_manifest.json",
  );
  const { loaded } = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const nodes = loaded.slice(OFFSET, OFFSET + LIMIT);

  await page.waitForFunction(
    () => window.app?.graph && window.LiteGraph?.createNode,
    { timeout: 120000 },
  );

  const auditOne = async (className) => {
    return page.evaluate(async (className) => {
      const lg = window.LiteGraph;
      const app = window.app;
      if (!lg?.createNode || !app?.graph) {
        return { className, ok: false, error: "no LiteGraph/app" };
      }
      let node;
      try {
        if (!lg.registered_node_types?.[className]) {
          return { className, ok: false, error: "type_not_registered" };
        }
        node = lg.createNode(className);
        if (!node) return { className, ok: false, error: "createNode_null" };
        app.graph.add(node, false);
        const n = (app.graph._nodes || []).length;
        node.pos = [60 + (n % 3) * 440, 60 + Math.floor(n / 3) * 280];
        const sz = node.computeSize?.() || [320, 180];
        node.setSize?.(sz);
        app.graph.setDirtyCanvas?.(true, true);
      } catch (e) {
        return { className, ok: false, error: String(e?.message || e) };
      }
      await new Promise((r) => setTimeout(r, 280));
      const el = node.el;
      const rect = el?.getBoundingClientRect?.();
      const errors = [];
      if (!rect || rect.height < 6) errors.push("zero_height");
      if (rect && rect.height > 1400) errors.push("oversized_h");
      if (className === "WanFaceController3DV2") {
        const ed = el?.querySelector?.(".fc3d-editor-root");
        const edR = ed?.getBoundingClientRect?.();
        if (!ed) errors.push("no_fc3d_editor");
        if (rect && edR && rect.height > 800 && edR.height < rect.height * 0.5) {
          errors.push("black_slab");
        }
        const vis = (node.widgets || []).filter((w) => {
          if (w.name === "face_overlay" || w.__fc3d_chrome_hidden) return false;
          return w.element && w.element.style.display !== "none" && !w.hidden;
        }).length;
        if (vis > 2) errors.push(`chrome_visible:${vis}`);
        const lastW = node.widgets?.[node.widgets.length - 1]?.name;
        if (lastW !== "face_overlay") errors.push(`dom_not_last:${lastW}`);
      }
      return {
        className,
        ok: errors.length === 0,
        errors,
        size: node.size ? [...node.size] : null,
        rect: rect ? { w: Math.round(rect.width), h: Math.round(rect.height) } : null,
      };
    }, className);
  };

  const results = [];
  for (const cn of nodes) {
    results.push(await auditOne(cn));
  }

  const outDir =
    "D:/PROJECT/Custom_Nodes/ComfyUI-WanAnimatePreprocessV2/scripts/_playwright_audit";
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `mcp_batch_${OFFSET}_${nodes.length}.json`);
  fs.writeFileSync(
    outFile,
    JSON.stringify(
      {
        offset: OFFSET,
        count: nodes.length,
        passed: results.filter((r) => r.ok).length,
        failed: results.filter((r) => !r.ok),
        results,
      },
      null,
      2,
    ),
  );

  // Face controller screenshot + tab click
  const fc = results.find((r) => r.className === "WanFaceController3DV2");
  if (fc) {
    await page.evaluate(() => {
      const app = window.app;
      const node = (app.graph._nodes || []).find(
        (n) => n.comfyClass === "WanFaceController3DV2",
      );
      if (node) app.canvas.selectNode(node);
    });
    await page.waitForTimeout(500);
    await page.screenshot({
      path: path.join(outDir, "WanFaceController3DV2.png"),
      fullPage: false,
    });
    await page.evaluate(() => {
      const root = document.querySelector(".fc3d-editor-root");
      const btn = root?.querySelectorAll?.("button");
      for (const b of btn || []) {
        if (b.textContent === "Expr") {
          b.click();
          break;
        }
      }
    });
    await page.waitForTimeout(400);
    await page.screenshot({
      path: path.join(outDir, "WanFaceController3DV2_expr_tab.png"),
      fullPage: false,
    });
  }

  return {
    file: outFile,
    tested: results.length,
    passed: results.filter((r) => r.ok).length,
    failed: results.filter((r) => !r.ok).map((r) => ({
      n: r.className,
      e: r.errors || r.error,
    })),
  };
};
