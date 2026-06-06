/**
 * Live ComfyUI node audit — adds each pack node via LiteGraph API, checks UI health.
 *
 * Usage (from repo root):
 *   npx playwright install chromium
 *   node ComfyUI-WanAnimatePreprocessV2/scripts/comfyui_playwright_node_audit.mjs
 *
 * Env:
 *   COMFY_URL=http://127.0.0.1:8188
 *   NODE_MANIFEST=path/to/_playwright_node_manifest.json
 */
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COMFY_URL = process.env.COMFY_URL || "http://127.0.0.1:8188";
const MANIFEST =
  process.env.NODE_MANIFEST ||
  path.join(__dirname, "_playwright_node_manifest.json");
const OUT_DIR = path.join(__dirname, "_playwright_audit");
const BATCH = Number(process.env.BATCH_SIZE || 0); // 0 = all
const OFFSET = Number(process.env.BATCH_OFFSET || 0);

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

async function waitForComfy(page) {
  await page.goto(COMFY_URL, { waitUntil: "load", timeout: 300000 });
  await page.waitForFunction(
    () => {
      const loading = document.querySelector('[status="Loading ComfyUI"], [aria-label="Loading ComfyUI"]');
      if (loading && loading.offsetParent !== null) return false;
      return typeof window.app !== "undefined" && window.app?.graph && window.LiteGraph?.createNode;
    },
    { timeout: 300000 },
  );
  await page.waitForTimeout(3000);
}

async function auditNode(page, className, displayHint) {
  return page.evaluate(
    async ({ className, displayHint }) => {
      const app = window.app;
      const lg = window.LiteGraph;
      if (!app?.graph || !lg?.createNode) {
        return { ok: false, error: "app.graph or LiteGraph.createNode missing" };
      }

      const info = app.graph?.extra?.nodeDefs?.[className] || lg.registered_node_types?.[className];
      if (!info && !lg.registered_node_types?.[className]) {
        return { ok: false, error: "node type not registered in frontend" };
      }

      let node;
      try {
        node = lg.createNode(className);
        if (!node) return { ok: false, error: "createNode returned null" };
        app.graph.add(node, false);
        const nodes = app.graph._nodes || [];
        const col = nodes.length % 4;
        const row = Math.floor(nodes.length / 4);
        node.pos = [40 + col * 420, 40 + row * 320];
        node.setSize?.(node.computeSize?.() || [320, 200]);
        app.graph.setDirtyCanvas?.(true, true);
        app.canvas?.draw?.(true, true);
      } catch (e) {
        return { ok: false, error: String(e?.message || e) };
      }

      await new Promise((r) => setTimeout(r, 400));

      const el = node.el;
      const rect = el?.getBoundingClientRect?.() || null;
      const widgets = (node.widgets || []).length;
      const domWidgets = (node.widgets || []).filter(
        (w) => w.element && w.name !== undefined,
      ).length;

      const blackSlab =
        className === "WanFaceController3DV2" &&
        rect &&
        rect.height > 900 &&
        (el?.querySelector?.(".fc3d-editor-root")?.getBoundingClientRect?.()
          ?.height || 0) <
          rect.height * 0.45;

      const huge =
        rect && (rect.height > 1200 || rect.width > 2000);

      const errors = [];
      if (!rect || rect.height < 8) errors.push("zero_height_node");
      if (huge) errors.push("oversized_node");
      if (blackSlab) errors.push("face_controller_black_slab");

      const visibleChrome = (node.widgets || []).filter((w) => {
        if (w.name === "face_overlay") return false;
        if (w.__fc3d_chrome_hidden) return false;
        const st = w.element?.style?.display;
        return w.element && st !== "none" && !w.hidden;
      }).length;

      if (className === "WanFaceController3DV2" && visibleChrome > 2) {
        errors.push(`face_visible_chrome_widgets:${visibleChrome}`);
      }

      return {
        ok: errors.length === 0,
        errors,
        widgets,
        domWidgets,
        size: node.size ? [...node.size] : null,
        rect: rect
          ? {
              w: Math.round(rect.width),
              h: Math.round(rect.height),
            }
          : null,
        title: node.title || displayHint || className,
        hasDomEditor: !!el?.querySelector?.(".fc3d-editor-root"),
      };
    },
    { className, displayHint },
  );
}

async function main() {
  if (!fs.existsSync(MANIFEST)) {
    console.error("Missing manifest. Run: python scripts/build_node_manifest.py");
    process.exit(1);
  }
  const { loaded } = JSON.parse(fs.readFileSync(MANIFEST, "utf8"));
  let nodes = loaded;
  if (BATCH > 0) nodes = nodes.slice(OFFSET, OFFSET + BATCH);

  ensureDir(OUT_DIR);
  const reportPath = path.join(OUT_DIR, `report_${OFFSET}_${nodes.length}.json`);
  const results = [];
  const consoleErrors = [];

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(String(err));
  });

  console.log(`ComfyUI audit: ${nodes.length} nodes @ ${COMFY_URL}`);
  await waitForComfy(page);

  for (let i = 0; i < nodes.length; i++) {
    const className = nodes[i];
    process.stdout.write(`[${i + 1}/${nodes.length}] ${className} ... `);
    let shot = null;
    try {
      const r = await auditNode(page, className, className);
      const slug = className.replace(/[^A-Za-z0-9_]/g, "_");
      shot = path.join(OUT_DIR, `${String(OFFSET + i).padStart(3, "0")}_${slug}.png`);
      if (!r.ok || className.includes("Face") || className.includes("Mask") || className.includes("Paint")) {
        await page.screenshot({ path: shot, fullPage: false });
      } else {
        shot = null;
      }
      results.push({ className, ...r, screenshot: shot });
      console.log(r.ok ? "OK" : `FAIL ${r.errors?.join(",")}`);
    } catch (e) {
      results.push({ className, ok: false, error: String(e) });
      console.log(`ERR ${e}`);
    }
  }

  const summary = {
    comfy_url: COMFY_URL,
    tested: results.length,
    passed: results.filter((r) => r.ok).length,
    failed: results.filter((r) => !r.ok),
    console_errors: [...new Set(consoleErrors)].slice(0, 50),
    results,
  };
  fs.writeFileSync(reportPath, JSON.stringify(summary, null, 2));
  console.log(`\nReport: ${reportPath}`);
  console.log(`Passed: ${summary.passed}/${summary.tested}`);
  if (summary.failed.length) {
    console.log("Failures:");
    for (const f of summary.failed) {
      console.log(`  - ${f.className}: ${f.errors?.join(",") || f.error}`);
    }
  }

  await browser.close();
  process.exit(summary.failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
