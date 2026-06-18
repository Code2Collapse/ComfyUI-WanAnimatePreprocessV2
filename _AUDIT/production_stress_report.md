# Production node stress test (2026-06-03)

## Scope

**WanAnimatePreprocessV2** registered production nodes:

| Node | Module |
|------|--------|
| WanIrisControlNetV2 | `iris_controlnet` |
| WanSHLightingTransferV2 | `sh_lighting` |
| WanQualityScorerJitterV2 | `quality_scorer_jitter` |
| WanPoseFormatConvertV2 | `pose_format_convert` |
| WanPoseDetectViTPoseV2 | `pose_detect_vitpose` |
| WanFaceController3DV2 | `face_controller_3d` |
| WanGazeETHXGazeV2 | `gaze_ethxgaze` (optional) |

**ComfyUI-CustomNodePacks** (feat/v2-ai-spine): MaskEditMEC, MaskMatting, WanDirector, etc.

## Runner

```text
ComfyUI-WanAnimatePreprocessV2/scripts/stress_production_nodes.py
```

Run inside the **same Python as ComfyUI** (needs `torch`, `numpy`):

```powershell
cd D:\PROJECT\Custom_Nodes\ComfyUI-WanAnimatePreprocessV2
& "PATH\TO\ComfyUI\python_embeded\python.exe" scripts\stress_production_nodes.py
```

Cursor shell only has minimal Python — import failed on `torch` here; that is an environment gap, not a node stub.

## WanFaceController3DV2 — findings (root cause of “static / corrupt”)

### Not a dummy

- `run()` applies real FACS basis math, head pose, gaze, blend, overrides on `POSEDATA`.
- Returns `preview_image`, `overlay_video`, `coeff_time_series_json`, `overlay_meta`.
- Live route `POST /c2c/fc3d_preview` re-runs `run()` for one frame when the graph has been queued once (`cache_put` on `unique_id`).

### Why it felt non-interactive

| Issue | Cause | Fix applied |
|-------|--------|-------------|
| Expression sliders do nothing visible | Canvas drew canonical landmarks; Expr tab only heat dots | Apply `FACS_BASIS` in `landmarksForFrame()` (same math as Python) |
| Drags miss landmarks when zoomed | Mouse coords ignored LiteGraph CSS scale | `eventCanvas()` zoom correction (points_bbox pattern) |
| “Corrupt” double UI | Only JSON hidden; 20+ float widgets still rendered below DOM widget | Hide all chrome widgets; mirror params in **Settings** tab |
| Node won’t resize | `computeLayoutSize.minWidth` + `setSize` on every ResizeObserver tick | Height-only `setSize`, `node.resizable = true`, `onResize` → redraw |
| Server truth lag | 300ms debounce + 412 until first queue | 120ms debounce; frame hint on 412 |

### Required user workflow

1. Connect **pose_data** (real POSEDATA bundle).
2. **Queue** the node once (fills preview cache + `overlay_meta` with per-frame `lms`).
3. Edit on **Face / Expr / Gaze / Pose** — canvas updates immediately; server preview follows after ~120ms.
4. Downstream must use this node’s **pose_data** output (not the untouched input).

## Branch note

- **ComfyUI-CustomNodePacks**: `feat/v2-ai-spine` (tracks `origin/feat/v2-ai-spine`).
- **ComfyUI-WanAnimatePreprocessV2**: local branch `feat/v2-ai-spine` created from current work; remote still only `main` — push when ready.

## Recommended verification in ComfyUI

1. Hard refresh (Ctrl+Shift+R).
2. New **Wan Face Controller 3D** node, wide enough to grab resize grip (bottom-right).
3. Queue with POSEDATA → Expr preset **Happy** → wireframe should smile on Face/Expr tabs.
4. `GET /c2c/fc3d_preview/status` or queue again if frame hint says “queue node once”.
