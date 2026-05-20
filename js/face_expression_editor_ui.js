// face_expression_editor_ui.js — in-canvas overlay for WanFaceExpressionEditorUI
//
// What this does
// --------------
// After the node executes, ComfyUI calls `onExecuted(message)` with the
// `ui.overlay_meta` payload produced by the Python side. We parse that JSON
// and render an inline canvas widget under the node showing:
//
//   • A square preview area sized to the actual node body width.
//   • The 68 iBUG-2D landmarks for the CURRENT frame (frame slider).
//   • Selected landmarks (mouth/brows/eyes/jaw — driven by the python
//     `selected` array) are drawn in MAUVE and are draggable.
//   • Unselected landmarks are drawn faintly so the user has context.
//   • Eye-emphasis landmarks (37/38/43/44) are drawn in PEACH so the user
//     understands why those move 1.5× faster.
//
// Drag semantics
// --------------
// Drag a mauve dot → coordinate is written into the `landmark_overrides_json`
// multiline STRING widget on the node, with the structure the python side
// expects:
//
//   { "frames": { "<frame>": { "<lm>": [x_norm, y_norm] } } }
//
// Re-queue the node and the override wins over the computed delta.
//
// Coordinates
// -----------
// The python `overlay_meta.frames[i].bbox` is `[x, y, w, h]` in pixel
// space of the source image. The python landmark-override contract is
// BBOX-NORMALISED, i.e. [0..1] inside that frame's bbox. We render the
// preview square as that bbox normalised to [0,1]×[0,1] inside the canvas,
// so screen→bbox-norm is just `(mx - canvasLeft) / canvasW`.
//
// The render does NOT need the original image — we draw the landmarks on
// a black/dark background with a faint grid. (The node already has an
// optional `reference_image` input but we don't have easy access to the
// pixel data here.) The dots are what matter for editing.
//
// Catppuccin-ish palette to match the rest of the pack's UI.

import { app } from "../../scripts/app.js";

const NODE_CLASS = "WanFaceExpressionEditorUI";
const OVERRIDE_WIDGET = "landmark_overrides_json";
const POSE_OVERRIDE_WIDGET = "pose_overrides_json";
const GAZE_OVERRIDE_WIDGET = "gaze_overrides_json";

const C = {
    bg:     "#1a1a22",
    border: "#2a2a35",
    grid:   "#26263a",
    dim:    "#6c7086",
    sel:    "#cba6f7",   // mauve — draggable face landmark
    emph:   "#fab387",   // peach — eye-emphasis
    other:  "#45475a",   // faint — context
    text:   "#cdd6f4",
    accent: "#89b4fa",
    // Pose-canvas palette (Slice 1).
    pose_joint:    "#a6e3a1",  // green — body joint dot
    pose_bone:     "#74c7ec",  // sky   — bone line
    pose_selected: "#f9e2af",  // yellow — joint being dragged
    pose_missing:  "#585b70",  // grey  — joint marker for missing joints
    // Gaze-handle palette (Slice 2).
    gaze_l:        "#f38ba8",  // pink  — left  eye handle / arrow
    gaze_r:        "#94e2d5",  // teal  — right eye handle / arrow
    gaze_drag:     "#f9e2af",  // yellow — handle being dragged
};

// OpenPose-18 joint names + a one-letter short label for cramped canvases.
const POSE18_NAMES = [
    "nose", "neck",
    "rShoulder", "rElbow", "rWrist",
    "lShoulder", "lElbow", "lWrist",
    "rHip", "rKnee", "rAnkle",
    "lHip", "lKnee", "lAnkle",
    "rEye", "lEye", "rEar", "lEar",
];
// Default fallback edges if python overlay_meta omits them (it shouldn't).
const POSE18_EDGES_DEFAULT = [
    [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
    [1,0],[0,14],[14,16],[0,15],[15,17],
];

// iBUG-68 polylines so the user sees the face shape, not a cloud of dots.
const SEGMENTS = [
    [0, 17],                  // jaw
    [17, 22], [22, 27],       // brows
    [27, 31], [31, 36],       // nose
    [36, 42, true],           // right eye (closed)
    [42, 48, true],           // left  eye (closed)
    [48, 60, true],           // outer mouth
    [60, 68, true],           // inner mouth
];

function readWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function parseOverrides(node) {
    const w = readWidget(node, OVERRIDE_WIDGET);
    if (!w || !w.value) return { frames: {} };
    try {
        const o = JSON.parse(w.value);
        if (!o.frames || typeof o.frames !== "object") o.frames = {};
        return o;
    } catch (_) {
        return { frames: {} };
    }
}

function writeOverrides(node, overrides) {
    const w = readWidget(node, OVERRIDE_WIDGET);
    if (!w) return;
    w.value = JSON.stringify(overrides, null, 0);
    // Trigger graph dirty so the new value is persisted into the workflow.
    if (w.callback) try { w.callback(w.value); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
}

// ── Pose-override helpers (Slice 1) ────────────────────────────────────
// Mirror of parseOverrides/writeOverrides but pointing at
// `pose_overrides_json`.  Coordinates here are IMAGE-normalised [0,1],
// not face-bbox-normalised.
function parsePoseOverrides(node) {
    const w = readWidget(node, POSE_OVERRIDE_WIDGET);
    if (!w || !w.value) return { frames: {} };
    try {
        const o = JSON.parse(w.value);
        if (!o.frames || typeof o.frames !== "object") o.frames = {};
        return o;
    } catch (_) {
        return { frames: {} };
    }
}
function writePoseOverrides(node, overrides) {
    const w = readWidget(node, POSE_OVERRIDE_WIDGET);
    if (!w) return;
    w.value = JSON.stringify(overrides, null, 0);
    if (w.callback) try { w.callback(w.value); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
}

// ── Gaze-override helpers (Slice 2) ────────────────────────────────────
// Mirror of parseOverrides/writeOverrides pointing at gaze_overrides_json.
// Per-frame payload: { l: [yaw_rad, pitch_rad], r: [yaw_rad, pitch_rad] }.
function parseGazeOverrides(node) {
    const w = readWidget(node, GAZE_OVERRIDE_WIDGET);
    if (!w || !w.value) return { frames: {} };
    try {
        const o = JSON.parse(w.value);
        if (!o.frames || typeof o.frames !== "object") o.frames = {};
        return o;
    } catch (_) {
        return { frames: {} };
    }
}
function writeGazeOverrides(node, overrides) {
    const w = readWidget(node, GAZE_OVERRIDE_WIDGET);
    if (!w) return;
    w.value = JSON.stringify(overrides, null, 0);
    if (w.callback) try { w.callback(w.value); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
}

function denormToCanvas(xn, yn, W, H) {
    return [xn * W, yn * H];
}
function canvasToNorm(cx, cy, W, H) {
    return [Math.max(0, Math.min(1, cx / W)), Math.max(0, Math.min(1, cy / H))];
}

// Reconstruct each frame's normalised landmarks from python output.
// overlay_meta.frames is [{i, ok, bbox?}, ...]; the landmark values aren't
// in there directly — we ship the d_norm delta. To render dots we need
// per-frame normalised landmarks, which the python overlay_meta does NOT
// include (it would balloon for long videos). So we render only the
// reference shape (centred) + the selected landmarks in their delta
// positions, allowing the user to drag from the predicted location.
//
// For now we render a CANONICAL iBUG-68 face (idealised positions) and
// apply the d_norm delta to give the user a visual cue. After the user
// runs the node once, frame 0 dots match the python output exactly.

// Canonical iBUG-68 in bbox-normalised coords (handcrafted neutral face).
// Source: derived from a centred, normalised mean-face shape. Coordinates
// are approximate; they're for VISUAL anchoring only — the actual numerical
// overrides written back are pure bbox-norm so they map back precisely.
const CANONICAL = [
    // jaw 0..16 (chin curve, slight smile)
    [0.05,0.45],[0.06,0.55],[0.08,0.65],[0.11,0.74],[0.16,0.82],
    [0.23,0.88],[0.32,0.93],[0.41,0.96],[0.50,0.97],[0.59,0.96],
    [0.68,0.93],[0.77,0.88],[0.84,0.82],[0.89,0.74],[0.92,0.65],
    [0.94,0.55],[0.95,0.45],
    // brows 17..21 (right brow)
    [0.15,0.30],[0.22,0.26],[0.30,0.25],[0.38,0.26],[0.45,0.30],
    // brows 22..26 (left brow)
    [0.55,0.30],[0.62,0.26],[0.70,0.25],[0.78,0.26],[0.85,0.30],
    // nose 27..30 (bridge)
    [0.50,0.36],[0.50,0.43],[0.50,0.49],[0.50,0.55],
    // nose 31..35 (nostrils)
    [0.43,0.60],[0.46,0.62],[0.50,0.63],[0.54,0.62],[0.57,0.60],
    // right eye 36..41
    [0.22,0.40],[0.27,0.37],[0.33,0.37],[0.38,0.40],[0.33,0.43],[0.27,0.43],
    // left eye 42..47
    [0.62,0.40],[0.67,0.37],[0.73,0.37],[0.78,0.40],[0.73,0.43],[0.67,0.43],
    // outer mouth 48..59
    [0.34,0.74],[0.40,0.71],[0.46,0.70],[0.50,0.71],[0.54,0.70],[0.60,0.71],
    [0.66,0.74],[0.60,0.79],[0.54,0.81],[0.50,0.82],[0.46,0.81],[0.40,0.79],
    // inner mouth 60..67
    [0.38,0.74],[0.45,0.74],[0.50,0.74],[0.55,0.74],[0.62,0.74],
    [0.55,0.76],[0.50,0.77],[0.45,0.76],
];

function buildOverlay(node) {
    if (node._faceOverlay) return node._faceOverlay;

    const root = document.createElement("div");
    root.style.cssText =
        "position:relative;width:100%;background:" + C.bg + ";" +
        "border-top:1px solid " + C.border + ";" +
        "border-radius:0 0 4px 4px;padding:6px;box-sizing:border-box;" +
        "display:flex;flex-direction:column;gap:4px;font:11px ui-sans-serif,system-ui;color:" + C.text + ";" +
        "pointer-events:auto;";

    // ── header strip: legend + frame slider
    const hdr = document.createElement("div");
    hdr.style.cssText = "display:flex;align-items:center;gap:8px;";
    const legend = document.createElement("span");
    legend.innerHTML =
        `<span style="color:${C.sel}">●</span> drag · ` +
        `<span style="color:${C.emph}">●</span> emph · ` +
        `<span style="color:${C.gaze_l}">↗</span>L · ` +
        `<span style="color:${C.gaze_r}">↗</span>R gaze · ` +
        `<span style="color:${C.other}">●</span> ctx`;
    const frameLbl = document.createElement("span");
    frameLbl.style.cssText = "margin-left:auto;color:" + C.dim;
    frameLbl.textContent = "frame 0 / 0";
    const reset = document.createElement("button");
    reset.textContent = "reset";
    reset.title = "Clear overrides for THIS frame";
    reset.style.cssText =
        "background:#2a2a35;color:" + C.text + ";border:1px solid " + C.border + ";" +
        "border-radius:3px;padding:2px 6px;cursor:pointer;font:11px ui-sans-serif;";
    hdr.append(legend, frameLbl, reset);
    root.appendChild(hdr);

    // ── canvas
    const cvs = document.createElement("canvas");
    cvs.width = 220;
    cvs.height = 220;
    cvs.style.cssText =
        "width:100%;height:auto;display:block;background:#0e0e16;" +
        "border:1px solid " + C.border + ";border-radius:4px;cursor:crosshair;";
    root.appendChild(cvs);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0"; slider.max = "0"; slider.value = "0";
    slider.style.cssText = "width:100%;margin:2px 0 0 0;accent-color:" + C.sel;
    root.appendChild(slider);

    // ── Pose canvas (Slice 1) ─────────────────────────────────────────
    // A second canvas drawn below the face canvas, rendering OpenPose-18
    // body joints in IMAGE-normalised space. Visibility is controlled by
    // a view-mode toggle: "face" | "pose" | "both".
    const viewBar = document.createElement("div");
    viewBar.style.cssText =
        "display:flex;align-items:center;gap:6px;margin-top:4px;font:11px ui-sans-serif;color:" + C.dim + ";";
    const viewLbl = document.createElement("span");
    viewLbl.textContent = "view:";
    const _mkViewBtn = (label, value) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.dataset.value = value;
        b.style.cssText =
            "background:#22222e;color:" + C.text + ";border:1px solid " + C.border + ";" +
            "border-radius:3px;padding:1px 6px;cursor:pointer;font:11px ui-sans-serif;";
        return b;
    };
    const btnFace = _mkViewBtn("face", "face");
    const btnPose = _mkViewBtn("pose", "pose");
    const btnBoth = _mkViewBtn("both", "both");
    const poseInfo = document.createElement("span");
    poseInfo.style.cssText = "margin-left:auto;color:" + C.dim + ";font:11px ui-monospace,monospace;";
    poseInfo.textContent = "—";
    viewBar.append(viewLbl, btnFace, btnPose, btnBoth, poseInfo);
    root.appendChild(viewBar);

    const poseCvs = document.createElement("canvas");
    poseCvs.width = 320; poseCvs.height = 240;
    poseCvs.style.cssText =
        "width:100%;height:auto;display:block;background:#0e0e16;" +
        "border:1px solid " + C.border + ";border-radius:4px;cursor:crosshair;margin-top:4px;";
    root.appendChild(poseCvs);

    // Hint when no pose data is available.
    const poseHint = document.createElement("div");
    poseHint.style.cssText = "font:11px ui-sans-serif;color:" + C.dim + ";margin-top:2px;";
    poseHint.textContent = "queue prompt once to enable pose editing";
    root.appendChild(poseHint);

    // ── Pose-canvas state ─────────────────────────────────────────────
    const pstate = {
        view:    "both",   // "face" | "pose" | "both"
        edges:   POSE18_EDGES_DEFAULT,
        names:   POSE18_NAMES,
        // Per-frame body data populated from overlay_meta.pose.frames.
        // Each entry: {ok:bool, w, h, kps:[[xn,yn]|null, ...]}.
        frames:  [],
        dragJ:   -1,
        hoverJ:  -1,
    };

    function _setView(v) {
        pstate.view = v;
        const showFace = (v === "face" || v === "both");
        const showPose = (v === "pose" || v === "both");
        cvs.style.display     = showFace ? "block" : "none";
        poseCvs.style.display = showPose ? "block" : "none";
        poseHint.style.display = (showPose && pstate.frames.length === 0) ? "block" : "none";
        for (const b of [btnFace, btnPose, btnBoth]) {
            const on = (b.dataset.value === v);
            b.style.background = on ? C.accent : "#22222e";
            b.style.color      = on ? "#11111a" : C.text;
        }
        drawPose();
    }
    btnFace.addEventListener("click", () => _setView("face"));
    btnPose.addEventListener("click", () => _setView("pose"));
    btnBoth.addEventListener("click", () => _setView("both"));

    // ── Pose drawing ──────────────────────────────────────────────────
    function _bodyKpsForFrame(f) {
        // Returns array of length 18: [xn, yn] or null per joint, after
        // applying any user override stored in pose_overrides_json.
        const fr = pstate.frames[f];
        if (!fr || !fr.ok || !Array.isArray(fr.kps)) return null;
        const out = fr.kps.map(p => Array.isArray(p) ? [p[0], p[1]] : null);
        const ov = parsePoseOverrides(node);
        const f_ov = ov.frames?.[String(f)];
        if (f_ov) {
            for (const [k, v] of Object.entries(f_ov)) {
                const i = parseInt(k, 10);
                if (i >= 0 && i < out.length && Array.isArray(v) && v.length === 2) {
                    out[i] = [Number(v[0]), Number(v[1])];
                }
            }
        }
        return out;
    }

    function drawPose() {
        const ctx = poseCvs.getContext("2d");
        const W = poseCvs.width, H = poseCvs.height;
        ctx.fillStyle = "#0e0e16";
        ctx.fillRect(0, 0, W, H);
        // Grid
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        for (let i = 1; i < 8; i++) {
            ctx.beginPath();
            ctx.moveTo((i / 8) * W, 0); ctx.lineTo((i / 8) * W, H); ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, (i / 6) * H); ctx.lineTo(W, (i / 6) * H); ctx.stroke();
        }

        if (pstate.frames.length === 0) {
            ctx.fillStyle = C.dim;
            ctx.font = "11px ui-sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("no pose data — run node once", W / 2, H / 2);
            poseInfo.textContent = "—";
            return;
        }

        const f = state.frame;
        const kps = _bodyKpsForFrame(f);
        if (!kps) {
            ctx.fillStyle = C.dim;
            ctx.font = "11px ui-sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("no body keypoints for frame " + f, W / 2, H / 2);
            poseInfo.textContent = `frame ${f}: no body`;
            return;
        }

        // Image aspect — fit-letterbox into canvas (preserves proportions
        // so the user sees the actual body shape, not stretched).
        const fr = pstate.frames[f];
        const imgW = Math.max(1, Number(fr.w) || 1);
        const imgH = Math.max(1, Number(fr.h) || 1);
        const scale = Math.min(W / imgW, H / imgH);
        const drawW = imgW * scale;
        const drawH = imgH * scale;
        const ox = (W - drawW) / 2;
        const oy = (H - drawH) / 2;

        // Letterbox frame outline.
        ctx.strokeStyle = C.border; ctx.lineWidth = 1;
        ctx.strokeRect(ox + 0.5, oy + 0.5, drawW - 1, drawH - 1);

        const _toCvs = (xn, yn) => [ox + xn * drawW, oy + yn * drawH];

        // Edges
        ctx.strokeStyle = C.pose_bone; ctx.lineWidth = 2;
        for (const [a, b] of pstate.edges) {
            const ka = kps[a], kb = kps[b];
            if (!ka || !kb) continue;
            const [x1, y1] = _toCvs(ka[0], ka[1]);
            const [x2, y2] = _toCvs(kb[0], kb[1]);
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        }

        // Joints
        for (let i = 0; i < kps.length; i++) {
            const k = kps[i];
            if (!k) continue;
            const [x, y] = _toCvs(k[0], k[1]);
            const isDrag = (pstate.dragJ === i);
            const isHover = (pstate.hoverJ === i);
            const r = isDrag ? 6 : (isHover ? 5 : 3.5);
            ctx.beginPath();
            ctx.fillStyle = isDrag ? C.pose_selected : C.pose_joint;
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
            // White stroke for visibility against bones.
            ctx.lineWidth = 1; ctx.strokeStyle = "#0e0e16";
            ctx.stroke();
        }

        // Hover label
        if (pstate.hoverJ >= 0 && kps[pstate.hoverJ]) {
            const [x, y] = _toCvs(kps[pstate.hoverJ][0], kps[pstate.hoverJ][1]);
            ctx.fillStyle = C.text;
            ctx.font = "10px ui-monospace,monospace";
            ctx.textAlign = "left";
            ctx.fillText(`${pstate.hoverJ}:${pstate.names[pstate.hoverJ] || "?"}`,
                         x + 7, y - 5);
        }

        // Counter showing how many joints have user overrides for this frame.
        const ov = parsePoseOverrides(node);
        const nOv = Object.keys(ov.frames?.[String(f)] || {}).length;
        poseInfo.textContent = `frame ${f}: ${kps.filter(k => k).length}/18 joints` + (nOv ? ` · ${nOv} overridden` : "");
    }

    // ── Pose canvas interaction ───────────────────────────────────────
    function _poseClientToCanvas(ev) {
        const r = poseCvs.getBoundingClientRect();
        const sx = poseCvs.width  / r.width;
        const sy = poseCvs.height / r.height;
        return [(ev.clientX - r.left) * sx, (ev.clientY - r.top) * sy];
    }

    function _pickJoint(mx, my) {
        if (pstate.frames.length === 0) return -1;
        const f = state.frame;
        const kps = _bodyKpsForFrame(f);
        if (!kps) return -1;
        const fr = pstate.frames[f];
        const imgW = Math.max(1, Number(fr.w) || 1);
        const imgH = Math.max(1, Number(fr.h) || 1);
        const W = poseCvs.width, H = poseCvs.height;
        const scale = Math.min(W / imgW, H / imgH);
        const drawW = imgW * scale, drawH = imgH * scale;
        const ox = (W - drawW) / 2, oy = (H - drawH) / 2;
        let best = -1, bestD = 10 * 10;  // 10px hit radius
        for (let i = 0; i < kps.length; i++) {
            if (!kps[i]) continue;
            const x = ox + kps[i][0] * drawW;
            const y = oy + kps[i][1] * drawH;
            const dx = mx - x, dy = my - y;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD) { bestD = d2; best = i; }
        }
        return best;
    }

    function _poseCanvasToNorm(cx, cy) {
        // Convert canvas pixel coords back to image-normalised [0,1].
        const f = state.frame;
        const fr = pstate.frames[f];
        if (!fr) return [0, 0];
        const imgW = Math.max(1, Number(fr.w) || 1);
        const imgH = Math.max(1, Number(fr.h) || 1);
        const W = poseCvs.width, H = poseCvs.height;
        const scale = Math.min(W / imgW, H / imgH);
        const drawW = imgW * scale, drawH = imgH * scale;
        const ox = (W - drawW) / 2, oy = (H - drawH) / 2;
        const xn = (cx - ox) / Math.max(1, drawW);
        const yn = (cy - oy) / Math.max(1, drawH);
        return [Math.max(0, Math.min(1, xn)), Math.max(0, Math.min(1, yn))];
    }

    poseCvs.addEventListener("mousedown", (ev) => {
        const [mx, my] = _poseClientToCanvas(ev);
        pstate.dragJ = _pickJoint(mx, my);
        if (pstate.dragJ >= 0) {
            ev.stopPropagation(); ev.preventDefault();
            drawPose();
        }
    });
    poseCvs.addEventListener("mousemove", (ev) => {
        const [mx, my] = _poseClientToCanvas(ev);
        if (pstate.dragJ >= 0) {
            const [xn, yn] = _poseCanvasToNorm(mx, my);
            const ov = parsePoseOverrides(node);
            const key = String(state.frame);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][String(pstate.dragJ)] = [
                Number(xn.toFixed(5)), Number(yn.toFixed(5)),
            ];
            writePoseOverrides(node, ov);
            drawPose();
            ev.stopPropagation();
        } else {
            const h = _pickJoint(mx, my);
            if (h !== pstate.hoverJ) { pstate.hoverJ = h; drawPose(); }
        }
    });
    const _endPoseDrag = () => {
        if (pstate.dragJ >= 0) { pstate.dragJ = -1; drawPose(); }
    };
    poseCvs.addEventListener("mouseup",    _endPoseDrag);
    poseCvs.addEventListener("mouseleave", () => { pstate.hoverJ = -1; _endPoseDrag(); drawPose(); });

    // Right-click on a joint = clear its override.
    poseCvs.addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        const [mx, my] = _poseClientToCanvas(ev);
        const j = _pickJoint(mx, my);
        if (j < 0) return;
        const ov = parsePoseOverrides(node);
        const key = String(state.frame);
        if (ov.frames?.[key]) {
            delete ov.frames[key][String(j)];
            if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
            writePoseOverrides(node, ov);
            drawPose();
        }
    });

    // ── state ──
    const state = {
        meta:    null,          // parsed overlay_meta
        frame:   0,
        dragLm:  -1,            // index of landmark being dragged
        hoverLm: -1,
    };

    // Gaze-handle state (Slice 2).
    const gstate = {
        dragEye:  null,         // "l" | "r" | null
        hoverEye: null,
    };

    function selectedSet() {
        return new Set(state.meta?.selected || []);
    }
    function emphSet() {
        return new Set(state.meta?.eye_emph || []);
    }

    // Compute landmarks for current frame: CANONICAL + d_norm * strength
    // for selected indices, plus any user override that already exists.
    function landmarksForFrame(f) {
        const lms = CANONICAL.map(p => [p[0], p[1]]);
        if (state.meta?.d_norm && Array.isArray(state.meta.d_norm)) {
            const sel = selectedSet();
            const stren = state.meta.strength ?? 0.7;
            for (let i = 0; i < lms.length && i < state.meta.d_norm.length; i++) {
                if (sel.has(i)) {
                    lms[i][0] += state.meta.d_norm[i][0] * stren;
                    lms[i][1] += state.meta.d_norm[i][1] * stren;
                }
            }
        }
        // Apply existing overrides for this frame.
        const ov = parseOverrides(node);
        const fr = ov.frames?.[String(f)];
        if (fr) {
            for (const [k, v] of Object.entries(fr)) {
                const i = parseInt(k, 10);
                if (i >= 0 && i < lms.length && Array.isArray(v) && v.length === 2) {
                    lms[i][0] = v[0];
                    lms[i][1] = v[1];
                }
            }
        }
        return lms;
    }

    function draw() {
        const ctx = cvs.getContext("2d");
        const W = cvs.width, H = cvs.height;
        ctx.fillStyle = "#0e0e16";
        ctx.fillRect(0, 0, W, H);

        // Faint grid (every 25%).
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            ctx.beginPath();
            ctx.moveTo((i / 4) * W, 0); ctx.lineTo((i / 4) * W, H); ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, (i / 4) * H); ctx.lineTo(W, (i / 4) * H); ctx.stroke();
        }

        if (!state.meta) {
            ctx.fillStyle = C.dim;
            ctx.font = "11px ui-sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("queue prompt once to enable editing", W / 2, H / 2);
            return;
        }

        const lms = landmarksForFrame(state.frame);
        const sel = selectedSet();
        const emp = emphSet();

        // Draw polylines (context).
        ctx.strokeStyle = C.other;
        ctx.lineWidth = 1;
        for (const seg of SEGMENTS) {
            const [a, b, closed] = seg;
            ctx.beginPath();
            for (let i = a; i < b; i++) {
                const [x, y] = denormToCanvas(lms[i][0], lms[i][1], W, H);
                if (i === a) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            if (closed) ctx.closePath();
            ctx.stroke();
        }

        // Draw dots.
        for (let i = 0; i < lms.length; i++) {
            const [x, y] = denormToCanvas(lms[i][0], lms[i][1], W, H);
            const isSel  = sel.has(i);
            const isEmph = emp.has(i);
            const r = (state.hoverLm === i || state.dragLm === i) ? 5
                    : isSel ? 3.2 : 2;
            ctx.beginPath();
            ctx.fillStyle = isEmph ? C.emph : (isSel ? C.sel : C.other);
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
        }

        // Show landmark index on hover.
        if (state.hoverLm >= 0) {
            const [x, y] = denormToCanvas(lms[state.hoverLm][0], lms[state.hoverLm][1], W, H);
            ctx.fillStyle = C.text;
            ctx.font = "10px ui-monospace,monospace";
            ctx.textAlign = "left";
            ctx.fillText(String(state.hoverLm), x + 6, y - 4);
        }

        // ── Gaze handles (Slice 2) ────────────────────────────────────
        // Anchor at the eye centroid (computed from the current frame's
        // FINAL face landmarks — which includes any face-override the user
        // already applied).  The tip is offset by the (yaw, pitch) override
        // (if any) or by the python-computed gaze (if any) for visual cue.
        const handles = computeGazeHandles(lms, W, H);
        for (const h of handles) {
            const isDrag = (gstate.dragEye === h.eye);
            const isHov  = (gstate.hoverEye === h.eye);
            const col    = isDrag ? C.gaze_drag : (h.eye === "l" ? C.gaze_l : C.gaze_r);
            // Anchor (eye centroid) — small ring.
            ctx.strokeStyle = col; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(h.ax, h.ay, 3, 0, Math.PI * 2); ctx.stroke();
            // Shaft.
            ctx.beginPath(); ctx.moveTo(h.ax, h.ay); ctx.lineTo(h.tx, h.ty); ctx.stroke();
            // Tip dot (the draggable handle).
            const r = isDrag ? 5.5 : (isHov ? 4.5 : 3.5);
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(h.tx, h.ty, r, 0, Math.PI * 2); ctx.fill();
        }
    }

    // ── Gaze handle math (Slice 2) ────────────────────────────────────
    // Eye-centroid: mean of iBUG-68 36..41 (right eye) / 42..47 (left eye).
    // The python overlay provides max_yaw_rad / max_pitch_rad; we map
    // (yaw, pitch) → screen offset linearly so the handle never escapes
    // a sensible radius around the eye.
    const _eyeRange = (eye) => (eye === "r") ? [36, 42] : [42, 48];
    function _eyeCentroid(lms, eye) {
        const [a, b] = _eyeRange(eye);
        let sx = 0, sy = 0, n = 0;
        for (let i = a; i < b; i++) { sx += lms[i][0]; sy += lms[i][1]; n++; }
        return [sx / n, sy / n];
    }
    // Pixel reach of the handle tip away from the eye, in fractions of W/H.
    // Values chosen so a maxed-out yaw lands about half-an-eye-width away
    // from the centroid, which matches how AE Face Director visualises it.
    const _GAZE_OFFSET_X_FRAC = 0.10;
    const _GAZE_OFFSET_Y_FRAC = 0.06;
    function _currentGazeForEye(f, eye) {
        // 1) explicit user override wins
        const ov = parseGazeOverrides(node);
        const fov = ov.frames?.[String(f)]?.[eye];
        if (Array.isArray(fov) && fov.length === 2) {
            return { yaw: Number(fov[0]) || 0, pitch: Number(fov[1]) || 0 };
        }
        // 2) python-computed gaze for this frame
        const gframes = state.meta?.gaze?.frames;
        const gf = Array.isArray(gframes) ? gframes[f] : null;
        const pair = gf && (eye === "l" ? gf.l_gaze : gf.r_gaze);
        if (Array.isArray(pair) && pair.length === 2) {
            return { yaw: Number(pair[0]) || 0, pitch: Number(pair[1]) || 0 };
        }
        return { yaw: 0, pitch: 0 };
    }
    function _gazeMax() {
        return {
            yaw:   Number(state.meta?.gaze?.max_yaw_rad)   || (30 * Math.PI / 180),
            pitch: Number(state.meta?.gaze?.max_pitch_rad) || (25 * Math.PI / 180),
        };
    }
    function computeGazeHandles(lms, W, H) {
        if (!state.meta?.gaze) return [];
        const max = _gazeMax();
        const out = [];
        for (const eye of ["l", "r"]) {
            const [cxN, cyN] = _eyeCentroid(lms, eye);
            const [ax, ay] = denormToCanvas(cxN, cyN, W, H);
            const { yaw, pitch } = _currentGazeForEye(state.frame, eye);
            const yawN   = Math.max(-1, Math.min(1, yaw   / max.yaw));
            const pitchN = Math.max(-1, Math.min(1, pitch / max.pitch));
            // image-Y is down → flip pitch sign so look-up moves the tip up
            const tx = ax + yawN   * _GAZE_OFFSET_X_FRAC * W;
            const ty = ay - pitchN * _GAZE_OFFSET_Y_FRAC * H;
            out.push({ eye, ax, ay, tx, ty });
        }
        return out;
    }
    function pickGazeHandle(mx, my) {
        if (!state.meta?.gaze) return null;
        const lms = landmarksForFrame(state.frame);
        const handles = computeGazeHandles(lms, cvs.width, cvs.height);
        let best = null, bestD = 10 * 10; // 10px hit radius
        for (const h of handles) {
            const dx = mx - h.tx, dy = my - h.ty;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD) { bestD = d2; best = h.eye; }
        }
        return best;
    }
    function canvasToGaze(mx, my, eye) {
        const lms = landmarksForFrame(state.frame);
        const [cxN, cyN] = _eyeCentroid(lms, eye);
        const [ax, ay] = denormToCanvas(cxN, cyN, cvs.width, cvs.height);
        const max = _gazeMax();
        const yawN   = (mx - ax) / Math.max(1, _GAZE_OFFSET_X_FRAC * cvs.width);
        const pitchN = -(my - ay) / Math.max(1, _GAZE_OFFSET_Y_FRAC * cvs.height);
        const yaw   = Math.max(-max.yaw,   Math.min(max.yaw,   yawN   * max.yaw));
        const pitch = Math.max(-max.pitch, Math.min(max.pitch, pitchN * max.pitch));
        return { yaw, pitch };
    }

    function pickLandmark(mx, my) {
        if (!state.meta) return -1;
        const lms = landmarksForFrame(state.frame);
        const sel = selectedSet();
        const W = cvs.width, H = cvs.height;
        let best = -1, bestD = 8 * 8; // 8px hit radius
        for (let i = 0; i < lms.length; i++) {
            if (!sel.has(i)) continue; // only selected are interactive
            const [x, y] = denormToCanvas(lms[i][0], lms[i][1], W, H);
            const dx = mx - x, dy = my - y;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD) { bestD = d2; best = i; }
        }
        return best;
    }

    function clientToCanvas(ev) {
        const r = cvs.getBoundingClientRect();
        const sx = cvs.width  / r.width;
        const sy = cvs.height / r.height;
        return [(ev.clientX - r.left) * sx, (ev.clientY - r.top) * sy];
    }

    cvs.addEventListener("mousedown", (ev) => {
        const [mx, my] = clientToCanvas(ev);
        // Gaze handles take priority over face landmarks so the user can
        // grab the handle even when it lands near an eye-region dot.
        const eye = pickGazeHandle(mx, my);
        if (eye) {
            gstate.dragEye = eye;
            ev.stopPropagation(); ev.preventDefault();
            draw();
            return;
        }
        state.dragLm = pickLandmark(mx, my);
        if (state.dragLm >= 0) {
            ev.stopPropagation();
            ev.preventDefault();
        }
    });
    cvs.addEventListener("mousemove", (ev) => {
        const [mx, my] = clientToCanvas(ev);
        if (gstate.dragEye) {
            const { yaw, pitch } = canvasToGaze(mx, my, gstate.dragEye);
            const ov = parseGazeOverrides(node);
            const key = String(state.frame);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][gstate.dragEye] = [
                Number(yaw.toFixed(5)), Number(pitch.toFixed(5)),
            ];
            writeGazeOverrides(node, ov);
            draw();
            ev.stopPropagation();
            return;
        }
        if (state.dragLm >= 0) {
            const [xn, yn] = canvasToNorm(mx, my, cvs.width, cvs.height);
            const ov = parseOverrides(node);
            const key = String(state.frame);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][String(state.dragLm)] = [
                Number(xn.toFixed(4)), Number(yn.toFixed(4)),
            ];
            writeOverrides(node, ov);
            draw();
            ev.stopPropagation();
        } else {
            const newHovEye = pickGazeHandle(mx, my);
            const h = newHovEye ? -1 : pickLandmark(mx, my);
            if (newHovEye !== gstate.hoverEye || h !== state.hoverLm) {
                gstate.hoverEye = newHovEye;
                state.hoverLm = h;
                draw();
            }
        }
    });
    const endDrag = () => {
        let dirty = false;
        if (state.dragLm >= 0) { state.dragLm = -1; dirty = true; }
        if (gstate.dragEye)    { gstate.dragEye = null; dirty = true; }
        if (dirty) draw();
    };
    cvs.addEventListener("mouseup",    endDrag);
    cvs.addEventListener("mouseleave", () => {
        state.hoverLm = -1;
        gstate.hoverEye = null;
        endDrag();
        draw();
    });

    // Right-click on a gaze handle = clear that eye's override for this frame.
    cvs.addEventListener("contextmenu", (ev) => {
        const [mx, my] = clientToCanvas(ev);
        const eye = pickGazeHandle(mx, my);
        if (!eye) return;
        ev.preventDefault();
        const ov = parseGazeOverrides(node);
        const key = String(state.frame);
        if (ov.frames?.[key]) {
            delete ov.frames[key][eye];
            if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
            writeGazeOverrides(node, ov);
            draw();
        }
    });

    slider.addEventListener("input", () => {
        state.frame = parseInt(slider.value, 10) || 0;
        frameLbl.textContent = `frame ${state.frame} / ${slider.max}`;
        draw();
        drawPose();
    });

    reset.addEventListener("click", () => {
        const ov = parseOverrides(node);
        delete ov.frames?.[String(state.frame)];
        writeOverrides(node, ov);
        // Also clear pose overrides for this frame.
        const pov = parsePoseOverrides(node);
        if (pov.frames?.[String(state.frame)]) {
            delete pov.frames[String(state.frame)];
            writePoseOverrides(node, pov);
        }
        // Also clear gaze overrides for this frame.
        const gov = parseGazeOverrides(node);
        if (gov.frames?.[String(state.frame)]) {
            delete gov.frames[String(state.frame)];
            writeGazeOverrides(node, gov);
        }
        draw();
        drawPose();
    });

    // Public API: feed in new overlay_meta after execution.
    const api = {
        root,
        update(meta) {
            state.meta = meta;
            // Update pose-canvas state from meta.pose (Slice 1).
            const pmeta = (meta && typeof meta === "object") ? meta.pose : null;
            if (pmeta && Array.isArray(pmeta.frames)) {
                pstate.frames = pmeta.frames;
                if (Array.isArray(pmeta.edges) && pmeta.edges.length) {
                    pstate.edges = pmeta.edges;
                }
                if (Array.isArray(pmeta.joint_names) && pmeta.joint_names.length) {
                    pstate.names = pmeta.joint_names;
                }
            } else {
                pstate.frames = [];
            }
            // Determine longest of face / pose frame counts to drive slider.
            const nFace = Array.isArray(meta?.frames) ? meta.frames.length : 0;
            const nPose = pstate.frames.length;
            const n = Math.max(nFace, nPose);
            slider.max = String(Math.max(0, n - 1));
            if (state.frame > Number(slider.max)) {
                state.frame = Number(slider.max);
                slider.value = String(state.frame);
            }
            frameLbl.textContent = `frame ${state.frame} / ${slider.max}`;
            poseHint.style.display =
                (pstate.view !== "face" && pstate.frames.length === 0) ? "block" : "none";
            draw();
            drawPose();
        },
    };

    // Default view = both, applied AFTER api is wired so _setView can call drawPose.
    _setView("both");

    draw();
    node._faceOverlay = api;
    return api;
}

app.registerExtension({
    name: "MEC.WanFaceExpressionEditorUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            const node = this;
            const overlay = buildOverlay(node);

            // Attach as a DOM widget so it lives in the node body.
            const w = node.addDOMWidget("face_overlay", "div", overlay.root, {
                getValue() { return ""; },
                setValue(v) {},
                serialize: false,
            });
            // Give it a stable min-height so the node doesn't collapse before
            // first execution. Height accounts for face canvas + pose canvas
            // + headers/slider/view-bar.
            w.computeSize = () => [node.size?.[0] || 320, 560];

            // Restore any prior overlay_meta cached on the workflow's exec data
            // (best-effort; not all sessions persist this).
            if (node._cachedOverlayMeta) {
                try { overlay.update(JSON.parse(node._cachedOverlayMeta)); } catch (_) {}
            }
        };

        const _executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            _executed?.apply(this, arguments);
            const raw = message?.overlay_meta?.[0];
            if (!raw) return;
            this._cachedOverlayMeta = raw;
            let meta;
            try { meta = JSON.parse(raw); } catch (_) { return; }
            if (this._faceOverlay) {
                this._faceOverlay.update(meta);
            }
        };
    },
});
