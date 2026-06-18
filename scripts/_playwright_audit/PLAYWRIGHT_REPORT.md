# ComfyUI Pack Nodes — Playwright Live Audit

**Date:** 2026-06-04  
**ComfyUI:** http://127.0.0.1:8188  
**Method:** Playwright MCP (live browser), LiteGraph `createNode` + `computeSize` per node

## Summary

| Metric | Count |
|--------|------:|
| Nodes in manifest (loaded in ComfyUI) | 96 |
| Batches run | 4 × 24 nodes |
| **Passed (instantiate + valid size)** | **95 / 96** |
| Failed | 1 |

## Failures

| Node | Issue | `computeSize` |
|------|--------|----------------|
| `WanDirectorC2C` | `oversized` — default height **2030px** (by design: large director panel) | `[820, 2030]` |

## Wan Face Controller 3D — live interaction

| Check | Result |
|-------|--------|
| Node creates | OK |
| `computeSize` in batch audit | OK (no black_slab / dom_not_last) |
| Tabs present | Face, Expr, Gaze, Pose, Settings |
| `face_overlay` last widget | OK |
| Chrome widgets hidden | 34 hidden |
| Node size after interact | `[400, 448]` |

Screenshots (Playwright MCP):

- `d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2_live.png`
- `d:/PROJECT/Custom_Nodes/.playwright-mcp/WanFaceController3DV2_centered.png`

**Note:** Comfy is running in **canvas (non-Vue) node mode** here (`LiteGraph.vueNodesMode === false`). DOM `getBoundingClientRect()` is often `null` until the node is centered on screen; layout checks use `node.size` from `computeSize`.

## Nodes not loaded in ComfyUI (17)

Defined in repo but absent from `/object_info` (optional deps / not installed):

`CoordinatePlotter`, `ForbiddenVision*`, `InpaintCropImproved`, `InpaintStitchImproved`, `LatentBuilder`, `LatentRefiner`, `MatAnyone`, `MatAnyone2`, `SamplerSchedulerSettings`, `SeCModelLoader`, `SeCVideoSegmentation`, `SolidColorBatched`

## How to re-run

```powershell
python ComfyUI-WanAnimatePreprocessV2/scripts/build_node_manifest.py
python ComfyUI-WanAnimatePreprocessV2/scripts/gen_playwright_audit_batches.py
# Then in Cursor: Playwright MCP → navigate http://127.0.0.1:8188 → run comfy_audit_b0.js … b3.js + comfy_face_interact.js
```

Or headless (after ComfyUI finished loading, ~60s):

```powershell
cd ComfyUI-WanAnimatePreprocessV2/scripts
node comfyui_playwright_node_audit.mjs
```

## Large DOM nodes (informational)

These registered with tall default sizes (not failures unless > 1600h):

- `MaskEditMEC` ~1422h
- `MaskOpsMEC` ~1074h
- `MaskTrackerMEC` ~936h
- `InpaintCropProMEC` ~892h
- `DepthPoseCannyCombinedV2` ~774h

Consider default size caps in a follow-up if UX requires.
