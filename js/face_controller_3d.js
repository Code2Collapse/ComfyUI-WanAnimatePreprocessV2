// face_controller_3d.js -- in-canvas viewer for WanFaceController3DV2
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
import { t as _t } from "./_c2c_i18n.js";
// Tolerant wrapper: returns fallback on any i18n failure so missing
// bundle / locale errors never break the overlay.
function T(key, fallback, vars) {
    try { return _t(key, fallback, vars); } catch (_) { return fallback ?? key; }
}

const NODE_CLASS = "WanFaceController3DV2";
const OVERRIDE_WIDGET = "landmark_overrides_json";
const POSE_OVERRIDE_WIDGET = "pose_overrides_json";
const GAZE_OVERRIDE_WIDGET = "gaze_overrides_json";

// ─── Theme palette (CSS-var backed; hex fallbacks if tokens absent) ──
const _C_FALLBACK = {
    bg:     "#1a1a22",
    border: "#2a2a35",
    grid:   "#26263a",
    dim:    "#6c7086",
    sel:    "#cba6f7",   // mauve — draggable face landmark
    emph:   "#fab387",   // peach — eye-emphasis
    other:  "#45475a",   // faint — context
    text:   "#cdd6f4",
    accent: "#89b4fa",
    pose_joint:    "#a6e3a1",
    pose_bone:     "#74c7ec",
    pose_selected: "#f9e2af",
    pose_missing:  "#585b70",
    gaze_l:        "#f38ba8",
    gaze_r:        "#94e2d5",
    gaze_drag:     "#f9e2af",
    canvas_bg:     "#0e0e16",
    input_bg:      "#1a1a23",
    btn_off_bg:    "#22222e",
    fg_inverse:    "#11111a",
    ok_bg:         "#2d3b22",
    err_bg:        "#3b2222",
    info_bg:       "#222b3b",
};
const _C_TOKEN = {
    bg:     "--c2c-bg",
    border: "--c2c-surface2",
    grid:   "--c2c-surface1",
    dim:    "--c2c-sub",
    sel:    "--c2c-violetSoft",
    emph:   "--c2c-yellow",
    other:  "--c2c-surface2",
    text:   "--c2c-fg",
    accent: "--c2c-blue",
    pose_joint:    "--c2c-green",
    pose_bone:     "--c2c-blue",
    pose_selected: "--c2c-yellow",
    pose_missing:  "--c2c-sub",
    gaze_l:        "--c2c-red",
    gaze_r:        "--c2c-green",
    gaze_drag:     "--c2c-yellow",
    canvas_bg:     "--c2c-crust",
    input_bg:      "--c2c-mantle",
    btn_off_bg:    "--c2c-surface0",
    fg_inverse:    "--c2c-crust",
    ok_bg:         "--c2c-greenDim",
    err_bg:        "--c2c-redDim",
    info_bg:       "--c2c-blueDim",
};
const C = new Proxy(_C_FALLBACK, {
    get(target, key) {
        const tok = _C_TOKEN[key];
        if (tok) {
            try {
                const v = getComputedStyle(document.documentElement).getPropertyValue(tok).trim();
                if (v) return v;
            } catch { /* token lookup failed; fall through to hex fallback */ }
        }
        return target[key];
    },
});

// OpenPose-18 joint names + a one-letter short label for cramped canvases.
const POSE18_NAMES = [
    "nose", "neck",
    "rShoulder", "rElbow", "rWrist",
    "lShoulder", "lElbow", "lWrist",
    "rHip", "rKnee", "rAnkle",
    "lHip", "lKnee", "lAnkle",
    "rEye", "lEye", "rEar", "lEar",
];

// iBUG-68 human-readable names. Used by the hover tooltip + numeric
// coord editor so the user knows exactly which point they are editing.
const LM_NAMES = (() => {
    const out = new Array(68);
    for (let i = 0;  i <= 16; i++) out[i] = `jaw_${i}`;
    for (let i = 17; i <= 21; i++) out[i] = `brow_r_${i-17}`;
    for (let i = 22; i <= 26; i++) out[i] = `brow_l_${i-22}`;
    for (let i = 27; i <= 30; i++) out[i] = `nose_bridge_${i-27}`;
    for (let i = 31; i <= 35; i++) out[i] = `nose_lower_${i-31}`;
    for (let i = 36; i <= 41; i++) out[i] = `eye_r_${i-36}`;
    for (let i = 42; i <= 47; i++) out[i] = `eye_l_${i-42}`;
    for (let i = 48; i <= 59; i++) out[i] = `mouth_outer_${i-48}`;
    for (let i = 60; i <= 67; i++) out[i] = `mouth_inner_${i-60}`;
    return out;
})();

// Default meta used BEFORE the user has queued the node once. With this
// in place the face canvas is always interactive — the user can drag a
// landmark or type an exact (x, y) and those overrides will be honoured
// on the next execution.  Pose/gaze stay dormant until real data arrives.
function makeDefaultMeta() {
    const zeros = Array.from({ length: 68 }, () => [0, 0]);
    const selected = Array.from({ length: 68 }, (_, i) => i);
    return {
        selected,
        eye_emph: [37, 38, 43, 44],
        d_norm:   zeros,
        frames:   [{ i: 0, ok: true }],
        strength: 1.0,
        pose: { format: "openpose_18", joint_names: POSE18_NAMES, edges: POSE18_EDGES_DEFAULT, frames: [] },
        gaze: null,
        _synthetic: true,
    };
}
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

// ── rAF-coalesced widget persistence (P1.2 real-time fix) ─────────────
// Previously every `mousemove` during a drag did:
//   1. JSON.stringify(overrides)
//   2. w.callback(...) (which may itself walk subgraphs)
//   3. node.setDirtyCanvas(true, true) — FULL LiteGraph redraw
// On complex graphs this caused visible lag ("Pose Editor not real-time —
// Queue required" was actually "edits don't paint until I stop moving").
// We now keep the in-memory override object on the node, do a cheap
// in-canvas draw() synchronously (so the dot follows the cursor at 60fps),
// and coalesce the JSON write + LiteGraph repaint into one rAF tick.
const _fc3dPending = new WeakMap(); // node -> { ov, pose, gaze, raf }
function _fc3dSchedulePersist(node) {
    let st = _fc3dPending.get(node);
    if (!st) { st = { ov: null, pose: null, gaze: null, raf: 0 }; _fc3dPending.set(node, st); }
    if (st.raf) return st;
    st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        const flush = (key, widgetName) => {
            const data = st[key];
            if (!data) return;
            st[key] = null;
            const w = readWidget(node, widgetName);
            if (!w) return;
            w.value = JSON.stringify(data, null, 0);
            if (w.callback) try { w.callback(w.value); } catch (_) {}
        };
        flush("ov",   OVERRIDE_WIDGET);
        flush("pose", POSE_OVERRIDE_WIDGET);
        flush("gaze", GAZE_OVERRIDE_WIDGET);
        node.setDirtyCanvas?.(true, true);
    });
    return st;
}

function parseOverrides(node) {
    const pending = _fc3dPending.get(node);
    if (pending && pending.ov) return pending.ov;
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
    if (!readWidget(node, OVERRIDE_WIDGET)) return;
    const st = _fc3dSchedulePersist(node);
    st.ov = overrides;
}

// ── Pose-override helpers (Slice 1) ────────────────────────────────────
// Mirror of parseOverrides/writeOverrides but pointing at
// `pose_overrides_json`.  Coordinates here are IMAGE-normalised [0,1],
// not face-bbox-normalised.
function parsePoseOverrides(node) {
    const pending = _fc3dPending.get(node);
    if (pending && pending.pose) return pending.pose;
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
    if (!readWidget(node, POSE_OVERRIDE_WIDGET)) return;
    const st = _fc3dSchedulePersist(node);
    st.pose = overrides;
}

// ── Gaze-override helpers (Slice 2) ────────────────────────────────────
// Mirror of parseOverrides/writeOverrides pointing at gaze_overrides_json.
// Per-frame payload: { l: [yaw_rad, pitch_rad], r: [yaw_rad, pitch_rad] }.
function parseGazeOverrides(node) {
    const pending = _fc3dPending.get(node);
    if (pending && pending.gaze) return pending.gaze;
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
    if (!readWidget(node, GAZE_OVERRIDE_WIDGET)) return;
    const st = _fc3dSchedulePersist(node);
    st.gaze = overrides;
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
        "background:" + C.border + ";color:" + C.text + ";border:1px solid " + C.border + ";" +
        "border-radius:3px;padding:2px 6px;cursor:pointer;font:11px ui-sans-serif;";
    hdr.append(legend, frameLbl, reset);
    root.appendChild(hdr);

    // ── canvas
    const cvs = document.createElement("canvas");
    cvs.width = 220;
    cvs.height = 220;
    cvs.style.cssText =
        "width:100%;height:auto;display:block;background:" + C.canvas_bg + ";" +
        "border:1px solid " + C.border + ";border-radius:4px;cursor:crosshair;";
    root.appendChild(cvs);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0"; slider.max = "0"; slider.value = "0";
    slider.style.cssText = "width:100%;margin:2px 0 0 0;accent-color:" + C.sel;
    root.appendChild(slider);

    // Make the canvas keyboard-focusable so users can step frames /
    // clear overrides without reaching for the slider. Discoverability:
    // outline ring on focus + tooltip lists shortcuts.
    cvs.tabIndex = 0;
    cvs.title =
        "click+drag landmarks · arrow ←/→ step frame · shift+arrow ±10 · " +
        "home/end first/last · R reset frame · delete clear hovered point · esc blur";
    cvs.style.outline = "none";
    cvs.addEventListener("focus", () => {
        cvs.style.boxShadow = "0 0 0 2px " + C.accent;
    });
    cvs.addEventListener("blur", () => {
        cvs.style.boxShadow = "none";
    });

    // ── Status line (hover readout + selection echo) ───────────────────
    // Shows the nearest landmark / joint / gaze-handle the mouse is over,
    // its iBUG name, and its current normalised coordinates. Always-on so
    // the user can read out a point's position without queueing first.
    const statusEl = document.createElement("div");
    statusEl.style.cssText =
        "margin-top:4px;padding:3px 6px;background:" + C.canvas_bg + ";" +
        "border:1px solid " + C.border + ";border-radius:3px;" +
        "font:11px ui-monospace,monospace;color:" + C.text + ";" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    statusEl.textContent = T("fc3d.status.hoverHint", "hover a point to inspect \u2014 drag to move \u2014 or type exact coords below");
    root.appendChild(statusEl);

    // ── Numeric coordinate editor ──────────────────────────────────────
    // Compact panel that lets the user enter EXACT normalised coordinates
    // for any face landmark / pose joint / gaze eye, then commit them
    // into the appropriate override JSON for the current frame.
    //
    // Layout (single row, wraps on narrow nodes):
    //   [target ▾] [idx] [x] [y] [Set] [Clear] [Pick]
    //
    // - target combo: face (iBUG-68) | pose (OP-18) | gaze L | gaze R
    // - idx        : landmark/joint index (hidden for gaze)
    // - x, y       : face/pose → 0..1 (image-norm), gaze → radians (yaw,pitch)
    // - Set        : write into landmark_/pose_/gaze_overrides_json for state.frame
    // - Clear      : remove that point's override for state.frame
    // - Pick       : copy the current hovered point into the editor
    const editor = document.createElement("div");
    editor.style.cssText =
        "display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:4px;" +
        "padding:4px;background:" + C.canvas_bg + ";border:1px solid " + C.border + ";" +
        "border-radius:3px;font:11px ui-sans-serif;color:" + C.text + ";";

    const _mkLbl = (txt) => {
        const s = document.createElement("span");
        s.textContent = txt;
        s.style.cssText = "color:" + C.dim + ";font-size:10px;";
        return s;
    };
    const _mkNumInput = (placeholder, step, min, max, width) => {
        const i = document.createElement("input");
        i.type = "number";
        i.step = String(step);
        if (min !== null) i.min = String(min);
        if (max !== null) i.max = String(max);
        i.placeholder = placeholder;
        i.style.cssText =
            "width:" + width + "px;padding:2px 4px;background:" + C.input_bg + ";color:" + C.text + ";" +
            "border:1px solid " + C.border + ";border-radius:2px;" +
            "font:11px ui-monospace,monospace;";
        return i;
    };
    const _mkBtn = (label, title, color) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.title = title;
        b.style.cssText =
            "background:" + (color || C.border) + ";color:" + C.text + ";" +
            "border:1px solid " + C.border + ";border-radius:3px;" +
            "padding:2px 8px;cursor:pointer;font:11px ui-sans-serif;";
        return b;
    };

    const tgtSel = document.createElement("select");
    tgtSel.title = T("fc3d.tip.target", "What to edit: face landmark, pose joint, or gaze eye");
    tgtSel.title = "What to edit: face landmark, pose joint, or gaze eye";
    tgtSel.style.cssText =
        "padding:2px 4px;background:" + C.input_bg + ";color:" + C.text + ";" +
        "border:1px solid " + C.border + ";border-radius:2px;font:11px ui-sans-serif;";
    for (const [v, t] of [
        ["face",   T("fc3d.target.face",  "face (iBUG-68)")],
        ["pose",   T("fc3d.target.pose",  "pose (OP-18)")],
        ["gaze-l", T("fc3d.target.gazeL", "gaze L (rad)")],
        ["gaze-r", T("fc3d.target.gazeR", "gaze R (rad)")],
    ]) {
        const o = document.createElement("option"); o.value = v; o.textContent = t; tgtSel.appendChild(o);
    }

    const idxInput = _mkNumInput("idx", 1, 0, 67, 44);
    idxInput.value = "30";
    const xInput   = _mkNumInput("x", 0.001, null, null, 64);
    const yInput   = _mkNumInput("y", 0.001, null, null, 64);
    const btnSet   = _mkBtn(T("fc3d.btn.set",   "Set"),   T("fc3d.tip.set",   "Write these coordinates as the override for the current frame."), C.ok_bg);
    const btnClear = _mkBtn(T("fc3d.btn.clear", "Clear"), T("fc3d.tip.clear", "Remove this point's override for the current frame."),            C.err_bg);
    const btnPick  = _mkBtn(T("fc3d.btn.pick",  "Pick"),  T("fc3d.tip.pick",  "Copy the currently-hovered or last-selected point into these fields."), C.info_bg);

    const nameTag = document.createElement("span");
    nameTag.style.cssText = "color:" + C.dim + ";font:10px ui-monospace,monospace;margin-left:auto;";
    nameTag.textContent = "—";

    editor.append(
        _mkLbl(T("fc3d.lbl.target", "target")), tgtSel,
        _mkLbl(T("fc3d.lbl.idx",    "idx")),    idxInput,
        _mkLbl(T("fc3d.lbl.x",      "x")),      xInput,
        _mkLbl(T("fc3d.lbl.y",      "y")),      yInput,
        btnSet, btnClear, btnPick,
        nameTag,
    );
    root.appendChild(editor);

    // ── editor helpers ─────────────────────────────────────────────────
    function _refreshEditorUI() {
        const t = tgtSel.value;
        const isGaze = (t === "gaze-l" || t === "gaze-r");
        idxInput.style.display = isGaze ? "none" : "";
        if (isGaze) {
            xInput.step = "0.01"; xInput.placeholder = T("fc3d.ph.yawRad",   "yaw rad");
            yInput.step = "0.01"; yInput.placeholder = T("fc3d.ph.pitchRad", "pitch rad");
        } else {
            xInput.step = "0.001"; xInput.placeholder = T("fc3d.ph.xNorm", "x 0..1");
            yInput.step = "0.001"; yInput.placeholder = T("fc3d.ph.yNorm", "y 0..1");
            const maxIdx = (t === "pose") ? 17 : 67;
            idxInput.max = String(maxIdx);
            if ((parseInt(idxInput.value, 10) || 0) > maxIdx) idxInput.value = "0";
        }
        _refreshNameTag();
    }
    function _refreshNameTag() {
        const t = tgtSel.value;
        if (t === "face") {
            const i = parseInt(idxInput.value, 10) || 0;
            nameTag.textContent = `face[${i}] ${LM_NAMES[i] || "?"}`;
        } else if (t === "pose") {
            const i = parseInt(idxInput.value, 10) || 0;
            nameTag.textContent = `pose[${i}] ${POSE18_NAMES[i] || "?"}`;
        } else {
            nameTag.textContent = (t === "gaze-l")
                ? T("fc3d.tag.gazeL", "gaze L  (yaw,pitch) in radians")
                : T("fc3d.tag.gazeR", "gaze R  (yaw,pitch) in radians");
        }
    }
    tgtSel.addEventListener("change", _refreshEditorUI);
    idxInput.addEventListener("input", _refreshNameTag);

    btnSet.addEventListener("click", () => {
        const t = tgtSel.value;
        const x = Number(xInput.value);
        const y = Number(yInput.value);
        if (!isFinite(x) || !isFinite(y)) {
            statusEl.textContent = T("fc3d.err.setNaN", "\u2717 Set: x or y is not a number");
            return;
        }
        const key = String(state.frame);
        if (t === "face") {
            const i = parseInt(idxInput.value, 10);
            if (!(i >= 0 && i <= 67)) { statusEl.textContent = T("fc3d.err.faceIdx", "\u2717 face idx must be 0..67"); return; }
            const xc = Math.max(0, Math.min(1, x));
            const yc = Math.max(0, Math.min(1, y));
            const ov = parseOverrides(node);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][String(i)] = [Number(xc.toFixed(4)), Number(yc.toFixed(4))];
            writeOverrides(node, ov);
            statusEl.textContent = `✓ face[${i}] ${LM_NAMES[i]} ← (${xc.toFixed(3)}, ${yc.toFixed(3)}) @ f${key}`;
        } else if (t === "pose") {
            const i = parseInt(idxInput.value, 10);
            if (!(i >= 0 && i <= 17)) { statusEl.textContent = T("fc3d.err.poseIdx", "\u2717 pose idx must be 0..17"); return; }
            const xc = Math.max(0, Math.min(1, x));
            const yc = Math.max(0, Math.min(1, y));
            const ov = parsePoseOverrides(node);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][POSE18_NAMES[i]] = [Number(xc.toFixed(4)), Number(yc.toFixed(4))];
            writePoseOverrides(node, ov);
            statusEl.textContent = `✓ pose[${i}] ${POSE18_NAMES[i]} ← (${xc.toFixed(3)}, ${yc.toFixed(3)}) @ f${key}`;
        } else {
            const eye = (t === "gaze-l") ? "l" : "r";
            // Soft-clamp to ±90° so a typo doesn't kill the override JSON.
            const cap = Math.PI / 2;
            const yaw   = Math.max(-cap, Math.min(cap, x));
            const pitch = Math.max(-cap, Math.min(cap, y));
            const ov = parseGazeOverrides(node);
            if (!ov.frames[key]) ov.frames[key] = {};
            ov.frames[key][eye] = [Number(yaw.toFixed(5)), Number(pitch.toFixed(5))];
            writeGazeOverrides(node, ov);
            statusEl.textContent = `✓ gaze ${eye.toUpperCase()} ← (yaw=${yaw.toFixed(3)}, pitch=${pitch.toFixed(3)}) @ f${key}`;
        }
        draw(); drawPose(); drawTimeline?.();
    });

    btnClear.addEventListener("click", () => {
        const t = tgtSel.value;
        const key = String(state.frame);
        if (t === "face") {
            const i = parseInt(idxInput.value, 10);
            const ov = parseOverrides(node);
            if (ov.frames?.[key]) {
                delete ov.frames[key][String(i)];
                if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
                writeOverrides(node, ov);
            }
            statusEl.textContent = `✓ cleared face[${i}] @ f${key}`;
        } else if (t === "pose") {
            const i = parseInt(idxInput.value, 10);
            const ov = parsePoseOverrides(node);
            if (ov.frames?.[key]) {
                delete ov.frames[key][POSE18_NAMES[i]];
                if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
                writePoseOverrides(node, ov);
            }
            statusEl.textContent = `✓ cleared pose[${i}] @ f${key}`;
        } else {
            const eye = (t === "gaze-l") ? "l" : "r";
            const ov = parseGazeOverrides(node);
            if (ov.frames?.[key]) {
                delete ov.frames[key][eye];
                if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
                writeGazeOverrides(node, ov);
            }
            statusEl.textContent = `✓ cleared gaze ${eye.toUpperCase()} @ f${key}`;
        }
        draw(); drawPose(); drawTimeline?.();
    });

    btnPick.addEventListener("click", () => {
        // Prefer the hover; fall back to selTarget.
        if (state.hoverLm >= 0) {
            tgtSel.value = "face";
            idxInput.value = String(state.hoverLm);
            const lms = landmarksForFrame(state.frame);
            xInput.value = lms[state.hoverLm][0].toFixed(4);
            yInput.value = lms[state.hoverLm][1].toFixed(4);
            _refreshEditorUI();
            return;
        }
        if (pstate?.hoverJ >= 0 && pstate.frames[state.frame]) {
            const fr = pstate.frames[state.frame];
            const kps = _get_body_kps_safe(fr);
            const j = pstate.hoverJ;
            if (kps && kps[j]) {
                tgtSel.value = "pose";
                idxInput.value = String(j);
                xInput.value = (Number(kps[j][0]) || 0).toFixed(4);
                yInput.value = (Number(kps[j][1]) || 0).toFixed(4);
                _refreshEditorUI();
            }
            return;
        }
        statusEl.textContent = T("fc3d.status.hoverFirst", "hover a point first, then click Pick");
    });

    // Safe body-keypoint accessor (handles either flat array or {keypoints_body:...} shape).
    function _get_body_kps_safe(fr) {
        if (!fr) return null;
        if (Array.isArray(fr.keypoints_body)) return fr.keypoints_body;
        if (Array.isArray(fr.kps_body))       return fr.kps_body;
        if (Array.isArray(fr.body))           return fr.body;
        return null;
    }

    _refreshEditorUI();

    // ── Timeline (Slice 3) ────────────────────────────────────────────
    // A compact strip showing per-frame edit indicators (face/pose/gaze).
    // Click a tick to jump there.  Shift-click to mark a range; range
    // actions in the timelineBar operate on the highlighted span only.
    const tl = document.createElement("canvas");
    tl.width = 480;
    tl.height = 34;
    tl.style.cssText =
        "width:100%;height:34px;display:block;background:" + C.canvas_bg + ";" +
        "border:1px solid " + C.border + ";border-radius:4px;margin-top:4px;cursor:pointer;";
    tl.title = T("fc3d.tip.timeline", "timeline \u00b7 click=jump \u00b7 shift-click=range \u00b7 right-click=clear that frame");
    root.appendChild(tl);

    const tlBar = document.createElement("div");
    tlBar.style.cssText =
        "display:flex;align-items:center;gap:6px;margin-top:2px;font:11px ui-sans-serif;color:" + C.dim + ";";
    const tlLegend = document.createElement("span");
    tlLegend.innerHTML =
        `<span style="color:${C.sel}">▮</span>face · ` +
        `<span style="color:${C.pose_joint}">▮</span>pose · ` +
        `<span style="color:${C.gaze_l}">▮</span>gaze`;
    const tlInfo = document.createElement("span");
    tlInfo.style.cssText = "margin-left:auto;font:11px ui-monospace,monospace;";
    tlInfo.textContent = "—";
    const _mkTlBtn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.title = title;
        b.style.cssText =
            "background:" + C.border + ";color:" + C.text + ";border:1px solid " + C.border + ";" +
            "border-radius:3px;padding:1px 6px;cursor:pointer;font:11px ui-sans-serif;";
        return b;
    };
    const btnPropagate  = _mkTlBtn(T("fc3d.btn.propOff", "\u0394 off"), T("fc3d.tip.prop", "\u0394 propagate: when ON, dragging a face landmark or pose joint applies that edit as a DELTA to every frame in the selected range (or all frames if none selected). Lets you pose/express once and broadcast it."));
    const btnClearRange = _mkTlBtn(T("fc3d.btn.clearRange", "clear range"), T("fc3d.tip.clearRange", "Clear all overrides in the highlighted range (shift-click to set range)."));
    const btnClearAll   = _mkTlBtn(T("fc3d.btn.clearAll",   "clear all"),   T("fc3d.tip.clearAll",   "Clear ALL overrides (face / pose / gaze / \u0394) across every frame."));
    const btnSelClear   = _mkTlBtn(T("fc3d.btn.selClear", "\u2715"), T("fc3d.tip.selClear", "Drop the current range selection."));
    btnSelClear.style.padding = "1px 5px";
    tlBar.append(tlLegend, btnPropagate, btnClearRange, btnClearAll, btnSelClear, tlInfo);
    root.appendChild(tlBar);

    // ── Δ-propagation state (edit one frame → broadcast delta to a range) ─
    // When `propagateMode` is on, a face/pose drag does NOT write a per-frame
    // absolute override; instead it accumulates a DELTA into a `ranges`
    // entry of the override JSON, which the python node applies across every
    // frame in [start,end] (end = -1 ⇒ to last frame).  Mirrors the existing
    // coefficient `ranges` contract but for raw landmark/joint XY.
    let propagateMode = false;

    // Timeline state (Slice 3).
    let _tlRafPending = false;
    const tlstate = {
        selA: -1,        // selection anchor (-1 = no selection)
        selB: -1,
        hover: -1,       // hovered frame index
    };
    function _tlRange() {
        if (tlstate.selA < 0 || tlstate.selB < 0) return null;
        return [Math.min(tlstate.selA, tlstate.selB),
                Math.max(tlstate.selA, tlstate.selB)];
    }
    function _frameCount() {
        return Math.max(0, Number(slider.max) + 1);
    }
    function _frameToX(f, W) {
        const n = Math.max(1, _frameCount());
        return (f / Math.max(1, n - 1)) * (W - 4) + 2;
    }
    function _xToFrame(x, W) {
        const n = Math.max(1, _frameCount());
        const t = (x - 2) / Math.max(1, W - 4);
        return Math.max(0, Math.min(n - 1, Math.round(t * Math.max(1, n - 1))));
    }
    // ── Δ-propagation helpers ──────────────────────────────────────────
    // Active broadcast range: the timeline selection if present, else
    // [0, -1] meaning "every frame" (python resolves -1 → last frame).
    function _propRange() {
        const r = _tlRange();
        return r ? [r[0], r[1]] : [0, -1];
    }
    // Find (or create) the ranges[] entry matching [s,e]; returns its delta map.
    function _ensureRangeEntry(ov, s, e) {
        if (!Array.isArray(ov.ranges)) ov.ranges = [];
        let entry = ov.ranges.find(en => en && Number(en.start) === s && Number(en.end) === e);
        if (!entry) { entry = { start: s, end: e, delta: {} }; ov.ranges.push(entry); }
        if (!entry.delta || typeof entry.delta !== "object") entry.delta = {};
        return entry;
    }
    // Accumulated Δ for landmark/joint `idx` on frame `f` across all ranges
    // that cover it.  Used by the renderers so the propagated edit is shown
    // on every covered frame (incl. the one being dragged).
    function _accumRangeDelta(ov, idx, f) {
        let dx = 0, dy = 0;
        const N = _frameCount();
        const ranges = Array.isArray(ov.ranges) ? ov.ranges : [];
        for (const en of ranges) {
            if (!en || !en.delta) continue;
            const s = Math.max(0, Number(en.start) || 0);
            let e = Number(en.end);
            if (!(e >= 0)) e = Math.max(0, N - 1);
            if (f < s || f > e) continue;
            const d = en.delta[String(idx)];
            if (Array.isArray(d) && d.length === 2) { dx += Number(d[0]) || 0; dy += Number(d[1]) || 0; }
        }
        return [dx, dy];
    }
    function _editFlagsForFrame(f) {
        // Returns {face, pose, gaze} booleans for frame f.
        const k = String(f);
        const fov = parseOverrides(node).frames?.[k];
        const pov = parsePoseOverrides(node).frames?.[k];
        const gov = parseGazeOverrides(node).frames?.[k];
        return {
            face: !!(fov && Object.keys(fov).length),
            pose: !!(pov && Object.keys(pov).length),
            gaze: !!(gov && Object.keys(gov).length),
        };
    }
    function drawTimeline() {
        const ctx = tl.getContext("2d");
        const W = tl.width, H = tl.height;
        ctx.fillStyle = C.canvas_bg;
        ctx.fillRect(0, 0, W, H);
        const n = _frameCount();
        if (n <= 0) {
            ctx.fillStyle = C.dim;
            ctx.font = "10px ui-sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("queue prompt to populate timeline", W / 2, H / 2 + 3);
            tlInfo.textContent = "—";
            return;
        }
        // Range highlight band.
        const range = _tlRange();
        if (range) {
            const [a, b] = range;
            const xa = _frameToX(a, W);
            const xb = _frameToX(b, W);
            ctx.fillStyle = "rgba(137,180,250,0.18)"; // accent w/ alpha
            ctx.fillRect(Math.min(xa, xb) - 1, 2, Math.abs(xb - xa) + 3, H - 4);
        }
        // Baseline.
        ctx.strokeStyle = C.border; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, H - 6.5); ctx.lineTo(W, H - 6.5); ctx.stroke();
        // Per-frame stack of indicator dots: face (top), pose (mid), gaze (bot).
        const rowY = [H * 0.25, H * 0.50, H * 0.75];
        const rowC = [C.sel,    C.pose_joint, C.gaze_l];
        let nFace = 0, nPose = 0, nGaze = 0;
        for (let f = 0; f < n; f++) {
            const x = _frameToX(f, W);
            const fl = _editFlagsForFrame(f);
            if (fl.face) { nFace++; ctx.fillStyle = rowC[0]; ctx.fillRect(x - 1, rowY[0] - 1.5, 2.5, 3); }
            if (fl.pose) { nPose++; ctx.fillStyle = rowC[1]; ctx.fillRect(x - 1, rowY[1] - 1.5, 2.5, 3); }
            if (fl.gaze) { nGaze++; ctx.fillStyle = rowC[2]; ctx.fillRect(x - 1, rowY[2] - 1.5, 2.5, 3); }
        }
        // Tick marks every ~10% of the timeline.
        ctx.strokeStyle = C.grid;
        const step = Math.max(1, Math.round(n / 10));
        for (let f = 0; f <= n - 1; f += step) {
            const x = _frameToX(f, W);
            ctx.beginPath();
            ctx.moveTo(x, H - 6); ctx.lineTo(x, H - 2); ctx.stroke();
        }
        // Hover marker.
        if (tlstate.hover >= 0 && tlstate.hover < n) {
            const x = _frameToX(tlstate.hover, W);
            ctx.strokeStyle = C.dim; ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x, 2); ctx.lineTo(x, H - 2); ctx.stroke();
            ctx.setLineDash([]);
        }
        // Playhead (current frame) — solid line on top.
        const xp = _frameToX(state.frame, W);
        ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xp, 2); ctx.lineTo(xp, H - 2); ctx.stroke();
        // Status text.
        let txt = `f${state.frame} · ${nFace}f/${nPose}p/${nGaze}g`;
        if (range) txt += ` · sel[${range[0]}..${range[1]}]`;
        tlInfo.textContent = txt;
    }

    function _tlClient(ev) {
        const r = tl.getBoundingClientRect();
        const sx = tl.width  / r.width;
        const sy = tl.height / r.height;
        return [(ev.clientX - r.left) * sx, (ev.clientY - r.top) * sy];
    }
    tl.addEventListener("click", (ev) => {
        if (_frameCount() <= 0) return;
        const [mx] = _tlClient(ev);
        const f = _xToFrame(mx, tl.width);
        if (ev.shiftKey) {
            // extend selection from existing anchor (or playhead).
            if (tlstate.selA < 0) tlstate.selA = state.frame;
            tlstate.selB = f;
        } else {
            // single-click: jump + drop selection.
            tlstate.selA = tlstate.selB = -1;
        }
        state.frame = f;
        slider.value = String(f);
        frameLbl.textContent = `frame ${f} / ${slider.max}`;
        _refreshPropBtn();
        draw(); drawPose(); drawTimeline();
    });
    tl.addEventListener("mousemove", (ev) => {
        if (_frameCount() <= 0) return;
        const [mx] = _tlClient(ev);
        const f = _xToFrame(mx, tl.width);
        if (f !== tlstate.hover) {
            tlstate.hover = f;
            drawTimeline();
        }
    });
    tl.addEventListener("mouseleave", () => {
        if (tlstate.hover !== -1) { tlstate.hover = -1; drawTimeline(); }
    });
    tl.addEventListener("contextmenu", (ev) => {
        // Right-click on a frame = clear ALL overrides for THAT frame.
        if (_frameCount() <= 0) return;
        ev.preventDefault();
        const [mx] = _tlClient(ev);
        const f = _xToFrame(mx, tl.width);
        _clearFrame(f);
        draw(); drawPose(); drawTimeline();
    });

    function _clearFrame(f) {
        const k = String(f);
        const o1 = parseOverrides(node);
        const o2 = parsePoseOverrides(node);
        const o3 = parseGazeOverrides(node);
        let touched = false;
        if (o1.frames?.[k]) { delete o1.frames[k]; writeOverrides(node, o1);     touched = true; }
        if (o2.frames?.[k]) { delete o2.frames[k]; writePoseOverrides(node, o2); touched = true; }
        if (o3.frames?.[k]) { delete o3.frames[k]; writeGazeOverrides(node, o3); touched = true; }
        return touched;
    }
    // Drop any ranges[] entries whose [start,end] intersects [a,b].
    function _stripRangesIntersecting(ov, a, b) {
        if (!Array.isArray(ov.ranges) || !ov.ranges.length) return false;
        const N = _frameCount();
        const before = ov.ranges.length;
        ov.ranges = ov.ranges.filter(en => {
            if (!en) return false;
            const s = Math.max(0, Number(en.start) || 0);
            let e = Number(en.end);
            if (!(e >= 0)) e = Math.max(0, N - 1);
            return e < a || s > b;   // keep only entries that do NOT overlap
        });
        return ov.ranges.length !== before;
    }
    btnClearRange.addEventListener("click", () => {
        const range = _tlRange();
        if (!range) { tlInfo.textContent = T("fc3d.info.selectRangeFirst", "select a range first (shift-click)"); return; }
        for (let f = range[0]; f <= range[1]; f++) _clearFrame(f);
        // Also drop Δ-propagation entries that overlap the cleared range.
        const ofa = parseOverrides(node);
        if (_stripRangesIntersecting(ofa, range[0], range[1])) writeOverrides(node, ofa);
        const opo = parsePoseOverrides(node);
        if (_stripRangesIntersecting(opo, range[0], range[1])) writePoseOverrides(node, opo);
        draw(); drawPose(); drawTimeline();
    });
    btnClearAll.addEventListener("click", () => {
        writeOverrides(node,     { frames: {} });
        writePoseOverrides(node, { frames: {} });
        writeGazeOverrides(node, { frames: {} });
        tlstate.selA = tlstate.selB = -1;
        draw(); drawPose(); drawTimeline();
    });
    btnSelClear.addEventListener("click", () => {
        tlstate.selA = tlstate.selB = -1;
        _refreshPropBtn();
        drawTimeline();
    });

    function _refreshPropBtn() {
        const r = _tlRange();
        if (!propagateMode) {
            btnPropagate.textContent = T("fc3d.btn.propOff", "\u0394 off");
            btnPropagate.style.background = C.border;
            btnPropagate.style.color = C.text;
            btnPropagate.style.fontWeight = "normal";
        } else {
            btnPropagate.textContent = r ? `Δ sel[${r[0]}..${r[1]}]` : "Δ all";
            btnPropagate.style.background = C.accent;
            btnPropagate.style.color = C.fg_inverse;
            btnPropagate.style.fontWeight = "bold";
        }
    }
    btnPropagate.addEventListener("click", () => {
        propagateMode = !propagateMode;
        _refreshPropBtn();
        statusEl.textContent = propagateMode
            ? "Δ propagate ON — drag a point to broadcast that edit across the active range"
            : "Δ propagate OFF — drags edit only the current frame";
    });

    // ── Pose canvas (Slice 1) ─────────────────────────────────────────
    // A second canvas drawn below the face canvas, rendering OpenPose-18
    // body joints in IMAGE-normalised space. Visibility is controlled by
    // a view-mode toggle: "face" | "pose" | "both".
    const viewBar = document.createElement("div");
    viewBar.style.cssText =
        "display:flex;align-items:center;gap:6px;margin-top:4px;font:11px ui-sans-serif;color:" + C.dim + ";";
    const viewLbl = document.createElement("span");
    viewLbl.textContent = T("fc3d.lbl.view", "view:");
    const _mkViewBtn = (label, value) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.dataset.value = value;
        b.style.cssText =
            "background:" + C.btn_off_bg + ";color:" + C.text + ";border:1px solid " + C.border + ";" +
            "border-radius:3px;padding:1px 6px;cursor:pointer;font:11px ui-sans-serif;";
        return b;
    };
    const btnFace = _mkViewBtn("face", T("fc3d.view.face", "face"));
    const btnPose = _mkViewBtn("pose", T("fc3d.view.pose", "pose"));
    const btnBoth = _mkViewBtn("both", T("fc3d.view.both", "both"));
    // 3D editor button — lazy-loads Three.js from a CDN with graceful
    // fallback to the existing 2D canvas if the load fails (offline,
    // CSP block, etc.). The 3D editor mounts a self-contained overlay
    // below the pose canvas and writes back to the head_tx/ty/tz +
    // yaw/pitch/roll widgets through callbacks.
    const btn3D = _mkViewBtn(
        T("fc3d.view.threeD", "3D\u2026"),
        "3d",
    );
    btn3D.title = T(
        "fc3d.tip.threeD",
        "Open a 3D head editor (loads Three.js on demand). Drag the gizmo to set head translation; slide yaw/pitch/roll to rotate.",
    );
    let _fc3dEditor = null;   // { destroy, refresh } once mounted
    let _fc3dHost   = null;   // host div for the overlay
    btn3D.addEventListener("click", async () => {
        // Toggle behaviour: a second click closes the editor.
        if (_fc3dEditor) {
            try { _fc3dEditor.destroy(); } catch (_) {}
            _fc3dEditor = null;
            if (_fc3dHost) { try { _fc3dHost.remove(); } catch (_) {} _fc3dHost = null; }
            btn3D.style.background = C.btn_off_bg;
            try { _persistSave?.(); } catch (_) {}
            return;
        }
        btn3D.style.background = C.btn_on_bg || C.accent;
        // Build a host container directly after the pose canvas / hint.
        _fc3dHost = document.createElement("div");
        _fc3dHost.style.cssText = "margin-top:6px;";
        root.appendChild(_fc3dHost);

        // Widget read/write helpers for the 6 head-pose widgets. The
        // python node declares these as widget names yaw_deg, pitch_deg,
        // roll_deg, head_tx, head_ty, head_tz; if any of them are missing
        // (older python node), the corresponding getter returns 0 and the
        // setter is a no-op so the editor still works.
        const _wRead = (name) => {
            const w = readWidget(node, name);
            return w ? Number(w.value) || 0 : 0;
        };
        const _wWrite = (name, v) => {
            const w = readWidget(node, name);
            if (!w) return;
            const num = Number(v);
            if (!Number.isFinite(num)) return;
            // Respect widget min/max if declared by the widget config.
            let clamped = num;
            if (w.options) {
                if (w.options.min !== undefined) clamped = Math.max(w.options.min, clamped);
                if (w.options.max !== undefined) clamped = Math.min(w.options.max, clamped);
            }
            if (w.value !== clamped) {
                w.value = clamped;
                try { w.callback?.(clamped, node, w); } catch (_) {}
                node.setDirtyCanvas?.(true, true);
            }
        };

        try {
            const mod = await import("./face_3d_editor.js");
            if (!_fc3dHost) return;  // user closed before load resolved
            _fc3dEditor = await mod.mount3DEditor(_fc3dHost, {
                theme: C,
                getLandmarks: () => {
                    try { return landmarksForFrame(state.frame); }
                    catch (_) { return null; }
                },
                getHeadPose: () => ({
                    yaw:   _wRead("head_yaw_deg"),
                    pitch: _wRead("head_pitch_deg"),
                    roll:  _wRead("head_roll_deg"),
                    tx:    _wRead("head_tx"),
                    ty:    _wRead("head_ty"),
                    tz:    _wRead("head_tz"),
                    // Phase 1.A extra DOF — Face Director real-time editor.
                    scale:      _wRead("head_scale"),
                    jaw:        _wRead("jaw_rot_deg"),
                    neck_yaw:   _wRead("neck_yaw_deg"),
                    neck_pitch: _wRead("neck_pitch_deg"),
                }),
                setHeadPose: (partial) => {
                    if (partial.yaw   !== undefined) _wWrite("head_yaw_deg",   partial.yaw);
                    if (partial.pitch !== undefined) _wWrite("head_pitch_deg", partial.pitch);
                    if (partial.roll  !== undefined) _wWrite("head_roll_deg",  partial.roll);
                    if (partial.tx    !== undefined) _wWrite("head_tx",   partial.tx);
                    if (partial.ty    !== undefined) _wWrite("head_ty",   partial.ty);
                    if (partial.tz    !== undefined) _wWrite("head_tz",   partial.tz);
                    if (partial.scale      !== undefined) _wWrite("head_scale",     partial.scale);
                    if (partial.jaw        !== undefined) _wWrite("jaw_rot_deg",    partial.jaw);
                    if (partial.neck_yaw   !== undefined) _wWrite("neck_yaw_deg",   partial.neck_yaw);
                    if (partial.neck_pitch !== undefined) _wWrite("neck_pitch_deg", partial.neck_pitch);
                    // (P1.D) drive the live-mirror + server-resync engine
                    // directly so gizmo drags feel instant even if the
                    // widget-callback chain is throttled.
                    try { _scheduleLocalMirror(); _scheduleServerResync(); } catch (_) {}
                },
                onClose: () => {
                    _fc3dEditor = null;
                    if (_fc3dHost) { try { _fc3dHost.remove(); } catch (_) {} _fc3dHost = null; }
                    btn3D.style.background = C.btn_off_bg;
                    try { _persistSave?.(); } catch (_) {}
                },
            });
            try { _persistSave?.(); } catch (_) {}
        } catch (err) {
            // Hard failure (network down, bad CDN, etc.) — leave a
            // readable note in the host and revert the toggle.
            if (_fc3dHost) {
                _fc3dHost.textContent =
                    T("fc3d.err.three", "3D editor unavailable: ") +
                    (err?.message || String(err));
                _fc3dHost.style.cssText =
                    "margin-top:6px;padding:4px 6px;color:#ff7070;" +
                    "background:" + C.canvas_bg + ";border:1px solid " + C.border + ";" +
                    "border-radius:3px;font:11px ui-monospace,monospace;";
            }
            btn3D.style.background = C.btn_off_bg;
        }
    });
    const poseInfo = document.createElement("span");
    poseInfo.style.cssText = "margin-left:auto;color:" + C.dim + ";font:11px ui-monospace,monospace;";
    poseInfo.textContent = "—";
    viewBar.append(viewLbl, btnFace, btnPose, btnBoth, btn3D, poseInfo);
    root.appendChild(viewBar);

    const poseCvs = document.createElement("canvas");
    poseCvs.width = 320; poseCvs.height = 240;
    poseCvs.style.cssText =
        "width:100%;height:auto;display:block;background:" + C.canvas_bg + ";" +
        "border:1px solid " + C.border + ";border-radius:4px;cursor:crosshair;margin-top:4px;";
    root.appendChild(poseCvs);

    // Hint when no pose data is available.
    const poseHint = document.createElement("div");
    poseHint.style.cssText = "font:11px ui-sans-serif;color:" + C.dim + ";margin-top:2px;";
    poseHint.textContent = T("fc3d.hint.queueFirst", "queue prompt once to enable pose editing");
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
            b.style.background = on ? C.accent : C.btn_off_bg;
            b.style.color      = on ? C.fg_inverse : C.text;
        }
        drawPose();
        try { _persistSave?.(); } catch (_) {}
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
        // Δ-propagation: add range deltas (image-norm) BEFORE per-frame pins.
        if (Array.isArray(ov.ranges) && ov.ranges.length) {
            for (let i = 0; i < out.length; i++) {
                if (!out[i]) continue;
                const [dx, dy] = _accumRangeDelta(ov, i, f);
                if (dx || dy) {
                    out[i] = [
                        Math.max(0, Math.min(1, out[i][0] + dx)),
                        Math.max(0, Math.min(1, out[i][1] + dy)),
                    ];
                }
            }
        }
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
        ctx.fillStyle = C.canvas_bg;
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
            ctx.lineWidth = 1; ctx.strokeStyle = C.canvas_bg;
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
            if (propagateMode) {
                const ref = _bodyKpsForFrame(state.frame)[pstate.dragJ] || [0, 0];
                pstate._dragRef = [ref[0], ref[1]];
                const [s, e] = _propRange();
                pstate._dragRange = [s, e];
                const ov = parsePoseOverrides(node);
                const entry = _ensureRangeEntry(ov, s, e);
                const cur = entry.delta[String(pstate.dragJ)];
                pstate._dragBase = (Array.isArray(cur) && cur.length === 2)
                    ? [Number(cur[0]) || 0, Number(cur[1]) || 0] : [0, 0];
            }
            ev.stopPropagation(); ev.preventDefault();
            drawPose();
        }
    });
    poseCvs.addEventListener("mousemove", (ev) => {
        const [mx, my] = _poseClientToCanvas(ev);
        if (pstate.dragJ >= 0) {
            const [xn, yn] = _poseCanvasToNorm(mx, my);
            const ov = parsePoseOverrides(node);
            if (propagateMode) {
                const [s, e] = pstate._dragRange || _propRange();
                const ref = pstate._dragRef || [xn, yn];
                const base = pstate._dragBase || [0, 0];
                const entry = _ensureRangeEntry(ov, s, e);
                entry.delta[String(pstate.dragJ)] = [
                    Number((base[0] + (xn - ref[0])).toFixed(5)),
                    Number((base[1] + (yn - ref[1])).toFixed(5)),
                ];
                writePoseOverrides(node, ov);
                drawPose();
                ev.stopPropagation();
                return;
            }
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
            if (h >= 0) {
                const fr = pstate.frames[state.frame];
                const kps = _get_body_kps_safe(fr);
                const xy = kps && kps[h] ? kps[h] : [0, 0];
                const nm = (pstate.names && pstate.names[h]) || POSE18_NAMES[h] || `j${h}`;
                statusEl.textContent =
                    `pose[${h}] ${nm}  =  (${Number(xy[0]).toFixed(3)}, ${Number(xy[1]).toFixed(3)})  ● draggable`;
            } else {
                statusEl.textContent =
                    `pose cursor (norm)  =  (${(mx/poseCvs.width).toFixed(3)}, ${(my/poseCvs.height).toFixed(3)})`;
            }
        }
    });
    const _endPoseDrag = () => {
        if (pstate.dragJ >= 0) { pstate.dragJ = -1; drawPose(); }
    };
    poseCvs.addEventListener("mouseup",    _endPoseDrag);
    poseCvs.addEventListener("mouseleave", () => {
        pstate.hoverJ = -1; _endPoseDrag(); drawPose();
        statusEl.textContent = T("fc3d.status.hoverHint", "hover a point to inspect \u2014 drag to move \u2014 or type exact coords below");
    });

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
        meta:    makeDefaultMeta(), // synthetic until first execution
        frame:   0,
        dragLm:  -1,            // index of landmark being dragged
        hoverLm: -1,
        // Selection target for the numeric coord editor:
        //   { kind: "face"|"pose", idx: <int> } | { kind: "gaze", eye: "l"|"r" } | null
        selTarget: { kind: "face", idx: 30 },  // start on nose tip
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
        // Δ-propagation: add range deltas (bbox-norm) BEFORE per-frame pins
        // so an explicit per-frame edit always wins (matches python order).
        if (Array.isArray(ov.ranges) && ov.ranges.length) {
            for (let i = 0; i < lms.length; i++) {
                const [dx, dy] = _accumRangeDelta(ov, i, f);
                if (dx || dy) { lms[i][0] += dx; lms[i][1] += dy; }
            }
        }
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
        // Schedule a timeline refresh (cheap, coalesced via rAF) so every
        // override edit reflects immediately in the per-frame strip.
        if (!_tlRafPending) {
            _tlRafPending = true;
            requestAnimationFrame(() => { _tlRafPending = false; drawTimeline(); });
        }
        const ctx = cvs.getContext("2d");
        const W = cvs.width, H = cvs.height;
        ctx.fillStyle = C.canvas_bg;
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

        if (state.meta?._synthetic) {
            // Tiny notice — the canvas is fully interactive, this just tells
            // the user it's a canonical face until they queue.
            ctx.fillStyle = C.dim;
            ctx.font = "10px ui-sans-serif";
            ctx.textAlign = "left";
            ctx.fillText("canonical face (queue to pull real landmarks)", 4, H - 4);
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

    // Hover-only nearest landmark — does NOT require selected-membership.
    // Used to populate the status line so the user can identify any iBUG
    // index even if it isn't draggable.
    function nearestLandmarkAny(mx, my) {
        if (!state.meta) return -1;
        const lms = landmarksForFrame(state.frame);
        const W = cvs.width, H = cvs.height;
        let best = -1, bestD = 14 * 14;
        for (let i = 0; i < lms.length; i++) {
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
            if (propagateMode) {
                // Capture the displayed reference position + the active range
                // entry's current delta so we can broadcast cursor motion as Δ.
                const ref = landmarksForFrame(state.frame)[state.dragLm];
                state._dragRef = [ref[0], ref[1]];
                const [s, e] = _propRange();
                state._dragRange = [s, e];
                const ov = parseOverrides(node);
                const entry = _ensureRangeEntry(ov, s, e);
                const cur = entry.delta[String(state.dragLm)];
                state._dragBase = (Array.isArray(cur) && cur.length === 2)
                    ? [Number(cur[0]) || 0, Number(cur[1]) || 0] : [0, 0];
            }
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
            if (propagateMode) {
                // Broadcast: write a Δ into the active range entry so the
                // dragged point tracks the cursor on every covered frame.
                const [s, e] = state._dragRange || _propRange();
                const ref = state._dragRef || [xn, yn];
                const base = state._dragBase || [0, 0];
                const entry = _ensureRangeEntry(ov, s, e);
                entry.delta[String(state.dragLm)] = [
                    Number((base[0] + (xn - ref[0])).toFixed(4)),
                    Number((base[1] + (yn - ref[1])).toFixed(4)),
                ];
                writeOverrides(node, ov);
                draw();
                ev.stopPropagation();
                return;
            }
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
            // For hover (status line), use the permissive picker so the
            // user can identify any iBUG-68 index, even non-selected ones.
            const hAny = newHovEye ? -1 : nearestLandmarkAny(mx, my);
            // For drag-hit highlight, keep the strict picker.
            const h    = newHovEye ? -1 : pickLandmark(mx, my);
            if (newHovEye !== gstate.hoverEye || h !== state.hoverLm) {
                gstate.hoverEye = newHovEye;
                state.hoverLm = h;
                draw();
            }
            // Status line readout — face / gaze.
            if (newHovEye) {
                statusEl.textContent = `gaze ${newHovEye.toUpperCase()}  (drag to set yaw/pitch — right-click to clear)`;
            } else if (hAny >= 0) {
                const lms = landmarksForFrame(state.frame);
                const [xn, yn] = lms[hAny];
                const isSel = selectedSet().has(hAny);
                const tag = isSel ? "● draggable" : "∘ read-only";
                statusEl.textContent =
                    `face[${hAny}] ${LM_NAMES[hAny]}  =  (${xn.toFixed(3)}, ${yn.toFixed(3)})   ${tag}`;
            } else {
                statusEl.textContent = `cursor (norm)  =  (${(mx/cvs.width).toFixed(3)}, ${(my/cvs.height).toFixed(3)})`;
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
        statusEl.textContent = T("fc3d.status.hoverHint", "hover a point to inspect \u2014 drag to move \u2014 or type exact coords below");
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
        drawTimeline();
        try { _persistSave?.(); } catch (_) {}
    });

    // Keyboard navigation on the focused canvas. Centralised here so the
    // slider, frame label and all three sub-canvases stay in sync.
    const _stepFrame = (delta) => {
        const max = parseInt(slider.max, 10) || 0;
        const next = Math.min(max, Math.max(0, state.frame + delta));
        if (next === state.frame) return;
        state.frame = next;
        slider.value = String(next);
        frameLbl.textContent = `frame ${state.frame} / ${slider.max}`;
        draw(); drawPose(); drawTimeline();
        try { _persistSave?.(); } catch (_) {}
    };
    const _gotoFrame = (idx) => {
        const max = parseInt(slider.max, 10) || 0;
        const next = Math.min(max, Math.max(0, idx));
        if (next === state.frame) return;
        state.frame = next;
        slider.value = String(next);
        frameLbl.textContent = `frame ${state.frame} / ${slider.max}`;
        draw(); drawPose(); drawTimeline();
        try { _persistSave?.(); } catch (_) {}
    };
    cvs.addEventListener("keydown", (ev) => {
        // Don't fight with text inputs that may bubble keyboard events.
        const tgt = ev.target;
        if (tgt && tgt !== cvs && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.tagName === "SELECT")) return;
        const big = ev.shiftKey ? 10 : 1;
        switch (ev.key) {
            case "ArrowLeft":
            case "ArrowDown":
                _stepFrame(-big); ev.preventDefault(); break;
            case "ArrowRight":
            case "ArrowUp":
                _stepFrame(+big); ev.preventDefault(); break;
            case "Home":
                _gotoFrame(0); ev.preventDefault(); break;
            case "End":
                _gotoFrame(parseInt(slider.max, 10) || 0); ev.preventDefault(); break;
            case "r":
            case "R":
                // Same effect as the header "reset" button: clear all
                // overrides on the current frame.
                reset.click(); ev.preventDefault(); break;
            case "Escape":
                cvs.blur(); ev.preventDefault(); break;
            case "Delete":
            case "Backspace": {
                // Clear the override for the hovered landmark, if any.
                if (state.hoverLm >= 0) {
                    const ov = parseOverrides(node);
                    const key = String(state.frame);
                    if (ov.frames?.[key]) {
                        delete ov.frames[key][String(state.hoverLm)];
                        if (Object.keys(ov.frames[key]).length === 0) delete ov.frames[key];
                        writeOverrides(node, ov);
                        draw();
                    }
                    ev.preventDefault();
                }
                break;
            }
        }
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

    // ── (P1.D) Live-preview engine ──────────────────────────────────
    // Two-tier responsiveness:
    //   (1) Local mirror: every DOF widget change schedules a rAF tick
    //       that re-applies the rigid 2D transforms (yaw/pitch/roll/
    //       scale/tx/ty/gaze) to a cached BASELINE meta. Feels instant.
    //   (2) Server resync: 300 ms after the last edit, POST the current
    //       widget state to /c2c/fc3d_preview for an authoritative
    //       single-frame recompute (includes jaw/neck rotation which
    //       depend on per-frame body anatomy). The response replaces the
    //       mirrored state with ground truth for the visible frame.
    //
    // The local mirror only approximates: it doesn't run jaw/neck/
    // expression-coeff math (those need region-specific knowledge that
    // lives in python). For those, the visible state stays at the
    // last server truth until the 300 ms re-sync lands.

    let _localBaseline = null;       // { faceFrames, poseFrames, widgetSnap }
    let _mirrorRaf = 0;
    let _resyncTimer = 0;
    let _resyncInFlight = false;
    let _resyncSeq = 0;
    const DOF_NAMES = [
        "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
        "head_tx", "head_ty", "head_tz",
        "head_scale", "jaw_rot_deg", "neck_yaw_deg", "neck_pitch_deg",
        "gaze_yaw_deg", "gaze_pitch_deg",
        "expression_strength", "expression_clamp", "blend_strength",
    ];

    function _readAllDOF() {
        const out = {};
        for (const k of DOF_NAMES) {
            const w = readWidget(node, k);
            out[k] = w ? Number(w.value) || 0 : 0;
        }
        // Also capture the string JSONs / mode combos so the server
        // re-sync runs the exact same pipeline as a real queue.
        for (const k of [
            "expression_coeffs_json", "head_pose_json", "gaze_json",
            "use_metas", "propagate_head", "propagate_gaze",
            "landmark_overrides_json", "pose_overrides_json", "gaze_overrides_json",
        ]) {
            const w = readWidget(node, k);
            if (w) out[k] = w.value;
        }
        for (const k of ["blend_mouth", "blend_brows", "blend_eyes", "blend_jaw"]) {
            const w = readWidget(node, k);
            if (w) out[k] = !!w.value;
        }
        return out;
    }

    function _deepCloneFrames(frames) {
        if (!Array.isArray(frames)) return [];
        return frames.map(f => {
            if (!f) return f;
            const c = { ...f };
            if (Array.isArray(f.kps)) c.kps = f.kps.map(p => (Array.isArray(p) ? p.slice() : p));
            return c;
        });
    }

    function _captureBaseline() {
        _localBaseline = {
            faceFrames: Array.isArray(state.meta?.frames)
                ? state.meta.frames.map(fr => fr ? { ...fr } : fr)
                : [],
            poseFrames: _deepCloneFrames(pstate.frames),
            // Original 68-point landmark arrays read from the freshly
            // delivered overlay_meta (these are post-edit, so we record
            // them as the new "zero" for incremental mirror math).
            faceLm: {},   // frameIdx → [[x,y]*68] in image-norm coords
            widgetSnap: _readAllDOF(),
        };
        const lmsFrames = state.meta?.frames;
        if (Array.isArray(lmsFrames)) {
            for (let i = 0; i < lmsFrames.length; i++) {
                try {
                    const lms = landmarksForFrame(i);
                    if (Array.isArray(lms) && lms.length === 68) {
                        _localBaseline.faceLm[i] = lms.map(p => p.slice());
                    }
                } catch (_) {}
            }
        }
    }

    function _mirrorTransform(frameIdx) {
        // Apply (current DOF − baseline DOF) as a 2D rigid transform to
        // baseline face landmarks + body keypoints for this frame and
        // mutate state.meta.frames[i].d_norm / pstate.frames[i].kps so
        // the existing draw() / drawPose() picks the change up.
        if (!_localBaseline) return;
        const base = _localBaseline.widgetSnap;
        const now  = _readAllDOF();
        const dYaw   = (now.head_yaw_deg   - base.head_yaw_deg)   * Math.PI / 180;
        const dPitch = (now.head_pitch_deg - base.head_pitch_deg) * Math.PI / 180;
        const dRoll  = (now.head_roll_deg  - base.head_roll_deg)  * Math.PI / 180;
        const dTx    =  now.head_tx        - base.head_tx;
        const dTy    =  now.head_ty        - base.head_ty;
        const dTz    =  now.head_tz        - base.head_tz;
        const rScale = (now.head_scale     || 1) / (base.head_scale || 1);
        const dNYaw  = (now.neck_yaw_deg   - base.neck_yaw_deg)   * Math.PI / 180;
        const dNPit  = (now.neck_pitch_deg - base.neck_pitch_deg) * Math.PI / 180;

        const cosR = Math.cos(dRoll), sinR = Math.sin(dRoll);
        // Approximate yaw / pitch as horizontal / vertical squashes
        // about the face centroid (mirrors python _apply_head_rotation
        // for small angles using the canonical-z = 0 assumption).
        const cosY = Math.cos(dYaw);
        const cosP = Math.cos(dPitch);
        // tz uses the same 1/(1+tz) perspective shrink as python.
        const tzClamp = Math.max(-0.75, Math.min(3.0, dTz));
        const zoomTz = 1.0 / (1.0 + tzClamp);

        // ── Face landmarks ──────────────────────────────────────────
        const baseLm = _localBaseline.faceLm[frameIdx];
        const faceFr = state.meta?.frames?.[frameIdx];
        if (baseLm && faceFr) {
            // Centroid of baseline landmarks.
            let cx = 0, cy = 0;
            for (const p of baseLm) { cx += p[0]; cy += p[1]; }
            cx /= baseLm.length; cy /= baseLm.length;
            // Compose: scale (rScale × zoomTz) → roll → yaw/pitch squash → translate
            const s = rScale * zoomTz;
            // Face-bbox width for tx/ty units (matches python convention).
            let mnX = 1, mxX = 0, mnY = 1, mxY = 0;
            for (const p of baseLm) {
                if (p[0] < mnX) mnX = p[0]; if (p[0] > mxX) mxX = p[0];
                if (p[1] < mnY) mnY = p[1]; if (p[1] > mxY) mxY = p[1];
            }
            const bbW = Math.max(1e-6, mxX - mnX);
            // d_norm is the per-landmark delta in face-bbox-normalised
            // units (overlay_meta convention). Compute new = T(base).
            const dNorm = new Array(68);
            const newLm = new Array(68);
            for (let i = 0; i < baseLm.length; i++) {
                let x = baseLm[i][0] - cx;
                let y = baseLm[i][1] - cy;
                // scale
                x *= s; y *= s;
                // roll (in-plane rotation)
                const rx = x * cosR - y * sinR;
                const ry = x * sinR + y * cosR;
                // yaw / pitch squash
                let nx = rx * cosY;
                let ny = ry * cosP;
                nx += cx + dTx * bbW;
                ny += cy + dTy * bbW;
                newLm[i] = [nx, ny];
                dNorm[i] = [(nx - baseLm[i][0]) / bbW,
                            (ny - baseLm[i][1]) / bbW];
            }
            // Patch the per-frame entry: writing the absolute coords as
            // "lms" plus the delta in d_norm covers both rendering paths
            // (full-frame redraw vs. delta-style overlay).
            faceFr.lms = newLm;
            faceFr.d_norm = dNorm;
        }

        // ── Body keypoints (neck rotation only) ─────────────────────
        const basePose = _localBaseline.poseFrames?.[frameIdx];
        const livePose = pstate.frames?.[frameIdx];
        if (basePose && livePose && Array.isArray(basePose.kps)
            && Array.isArray(livePose.kps) && livePose.kps.length === 18) {
            const neck = basePose.kps[1];
            const baseKps = basePose.kps;
            if (Array.isArray(neck) && neck.length >= 2
                && (Math.abs(dNYaw) > 1e-3 || Math.abs(dNPit) > 1e-3)) {
                const nx = neck[0], ny = neck[1];
                const headCluster = [0, 14, 15, 16, 17];
                const cyN = Math.cos(dNYaw);
                const cpN = Math.cos(dNPit), spN = Math.sin(dNPit);
                const newKps = baseKps.map(p => (Array.isArray(p) ? p.slice() : p));
                for (const idx of headCluster) {
                    const p = baseKps[idx];
                    if (!Array.isArray(p)) continue;
                    const dx = p[0] - nx, dy = p[1] - ny;
                    newKps[idx][0] = nx + dx * cyN;
                    newKps[idx][1] = ny + dy * cpN - spN * Math.abs(dy);
                }
                livePose.kps = newKps;
            } else {
                // Reset to baseline if neck deltas are zero.
                livePose.kps = baseKps.map(p => (Array.isArray(p) ? p.slice() : p));
            }
        }
    }

    function _scheduleLocalMirror() {
        if (_mirrorRaf) return;
        _mirrorRaf = requestAnimationFrame(() => {
            _mirrorRaf = 0;
            try { _mirrorTransform(state.frame); } catch (_) {}
            try { draw(); drawPose(); } catch (_) {}
        });
    }

    async function _serverResync() {
        const seq = ++_resyncSeq;
        if (_resyncInFlight) return;  // a fresh trigger after timer fires already
        _resyncInFlight = true;
        try {
            const body = { node_id: String(node.id), frame_idx: state.frame, ..._readAllDOF() };
            const resp = await fetch("/c2c/fc3d_preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (seq !== _resyncSeq) return; // a newer request was already queued
            if (resp.status === 412) {
                // Cache miss — node hasn't been queued yet; silently skip.
                return;
            }
            if (!resp.ok) return;
            const data = await resp.json();
            if (seq !== _resyncSeq) return;
            // Splice the authoritative per-frame data into state.meta + pstate.
            const fIdx = data.frame_idx;
            const faceArr = data.face_norm;     // 68-point bbox-norm
            const bodyArr = data.body_kps;      // 18-point image-norm or null
            // Convert face-bbox-norm back into IMAGE-norm so it matches the
            // baseline coordinate system used by draw().
            if (Array.isArray(faceArr) && _localBaseline?.faceLm?.[fIdx]) {
                const baseLm = _localBaseline.faceLm[fIdx];
                let mnX = 1, mxX = 0, mnY = 1, mxY = 0;
                for (const p of baseLm) {
                    if (p[0] < mnX) mnX = p[0]; if (p[0] > mxX) mxX = p[0];
                    if (p[1] < mnY) mnY = p[1]; if (p[1] > mxY) mxY = p[1];
                }
                const bbW = Math.max(1e-6, mxX - mnX);
                const bbH = Math.max(1e-6, mxY - mnY);
                const absLm = faceArr.map(p => [mnX + p[0] * bbW, mnY + p[1] * bbH]);
                const faceFr = state.meta?.frames?.[fIdx];
                if (faceFr) {
                    faceFr.lms = absLm;
                    faceFr.d_norm = absLm.map((p, i) => [
                        (p[0] - baseLm[i][0]) / bbW,
                        (p[1] - baseLm[i][1]) / bbH,
                    ]);
                }
            }
            if (Array.isArray(bodyArr) && pstate.frames?.[fIdx]) {
                pstate.frames[fIdx].kps = bodyArr.map(p =>
                    Array.isArray(p) ? p.slice() : [NaN, NaN]);
            }
            try { draw(); drawPose(); } catch (_) {}
        } catch (_) {
            // Network/CORS failure — silent; local mirror still visible.
        } finally {
            _resyncInFlight = false;
        }
    }

    function _scheduleServerResync() {
        if (_resyncTimer) clearTimeout(_resyncTimer);
        _resyncTimer = setTimeout(() => {
            _resyncTimer = 0;
            _serverResync();
        }, 300);
    }

    // Hook callbacks on every DOF widget so any edit (slider, numeric
    // type-in, or the 3D gizmo via _wWrite) triggers both passes.
    function _hookDOFWidgets() {
        for (const k of DOF_NAMES) {
            const w = readWidget(node, k);
            if (!w || w.__fc3d_hooked) continue;
            const orig = w.callback;
            w.callback = function (v) {
                try { orig?.call(this, v, node, w); } catch (_) {}
                _scheduleLocalMirror();
                _scheduleServerResync();
            };
            w.__fc3d_hooked = true;
        }
    }
    // Defer one tick so widgets exist (they're added by ComfyUI after
    // onNodeCreated runs).
    setTimeout(_hookDOFWidgets, 0);
    setTimeout(_hookDOFWidgets, 250);

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
            // (P1.D) Capture this as the local-mirror baseline so subsequent
            // gizmo / widget edits can be rendered instantly relative to it
            // until the 300 ms server re-sync replaces it with truth.
            _captureBaseline();
            draw();
            drawPose();
            drawTimeline();
        },
        // Imperative control surface for the Ctrl+K command palette and
        // any other extension wanting to script the overlay.
        focus()       { try { cvs.focus(); } catch (_) {} },
        resetFrame()  { try { reset.click(); } catch (_) {} },
        gotoFirst()   { _gotoFrame(0); },
        gotoLast()    { _gotoFrame(parseInt(slider.max, 10) || 0); },
        step(delta)   { _stepFrame(delta); },
        setView(v)    { try { _setView(v); } catch (_) {} },
        // (P1.D) Live-preview control surface.
        mirrorRefresh() { try { _scheduleLocalMirror(); } catch (_) {} },
        serverResync()  { try { _scheduleServerResync(); } catch (_) {} },
        resetTransform() {
            for (const k of DOF_NAMES) {
                const w = readWidget(node, k);
                if (!w) continue;
                let def = 0;
                if (k === "head_scale" || k === "expression_strength" || k === "blend_strength") def = 1;
                if (k === "expression_clamp") def = w.options?.default ?? 1.5;
                if (w.value !== def) {
                    w.value = def;
                    try { w.callback?.(def, node, w); } catch (_) {}
                }
            }
            node.setDirtyCanvas?.(true, true);
            _scheduleLocalMirror();
            _scheduleServerResync();
        },
        // Tear down the 3D editor (if open) and release its WebGL/RAF
        // resources. Called by onRemoved so deleting a node doesn't leak
        // a GL context or an animation loop.
        destroy3D() {
            try { _fc3dEditor?.destroy(); } catch (_) {}
            _fc3dEditor = null;
            if (_fc3dHost) { try { _fc3dHost.remove(); } catch (_) {} _fc3dHost = null; }
        },
    };

    // ── (P1.E) Per-node UI persistence ─────────────────────────────
    // Persist transient UI state (view mode, current frame, 3D-editor
    // open/closed) keyed by node.id under localStorage, so reopening a
    // workflow brings back the user's last layout without affecting
    // ComfyUI's own widget-value persistence.
    const _persistKey = () => `mec.fc3d.overlay.${node.id ?? "_"}`;
    let _persistRaf = 0;
    function _persistSave() {
        if (_persistRaf) return;
        _persistRaf = requestAnimationFrame(() => {
            _persistRaf = 0;
            try {
                const snap = {
                    view: pstate.view,
                    frame: state.frame,
                    editor3d: !!_fc3dEditor,
                };
                localStorage.setItem(_persistKey(), JSON.stringify(snap));
            } catch (_) {}
        });
    }
    function _persistLoad() {
        try {
            const raw = localStorage.getItem(_persistKey());
            if (!raw) return null;
            return JSON.parse(raw);
        } catch (_) { return null; }
    }

    // Default view = both, applied AFTER api is wired so _setView can call drawPose.
    const _saved = _persistLoad();
    _setView(_saved?.view === "face" || _saved?.view === "pose" || _saved?.view === "both"
        ? _saved.view : "both");
    if (_saved && Number.isFinite(_saved.frame)) {
        const f = Math.max(0, Number(_saved.frame) | 0);
        state.frame = f;
        try { slider.value = String(f); } catch (_) {}
        frameLbl.textContent = `frame ${state.frame} / ${slider.max}`;
    }
    if (_saved?.editor3d) {
        // Re-open the 3D editor a tick later so the host DOM + overlay
        // are fully initialised first.
        setTimeout(() => { try { btn3D.click(); } catch (_) {} }, 0);
    }

    draw();
    drawTimeline();
    node._faceOverlay = api;
    return api;
}

app.registerExtension({
    name: "MEC.WanFaceController3DV2",

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
            // + headers/slider/view-bar. Use the `width` LiteGraph passes
            // (inset widget-column width) rather than `node.size[0]` to
            // avoid dark gutters bleeding through on both edges.
            w.computeSize = (width) => [width, 560];

            // Restore any prior overlay_meta cached on the workflow's exec data
            // (best-effort; not all sessions persist this).
            if (node._cachedOverlayMeta) {
                try { overlay.update(JSON.parse(node._cachedOverlayMeta)); } catch (_) {}
            }
            _fc3dInstances.add(node);
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

        const _removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._faceOverlay?.destroy3D?.(); } catch (_) {}
            _fc3dInstances.delete(this);
            return _removed?.apply(this, arguments);
        };
    },
});

// ── Ctrl+K command-palette integration ──────────────────────────────
// Tracks every live WanFaceController3DV2 node so the global palette
// can dispatch keyboard shortcuts at the most-recently-created one.
const _fc3dInstances = new Set();
function _fc3dActive() {
    let last = null;
    for (const n of _fc3dInstances) last = n;
    return last?._faceOverlay || null;
}
function _fc3dRegisterActions() {
    const reg = window.__C2C_ACTIONS__?.register;
    if (typeof reg !== "function") return;
    const enabled = () => _fc3dInstances.size > 0;
    const actions = [
        { id: "mec.faceController.focus",      title: "Face Controller: Focus canvas (enable arrow-key nav)", icon: "◎", keywords: ["face","focus","keyboard"],          run: () => _fc3dActive()?.focus()       },
        { id: "mec.faceController.resetFrame", title: "Face Controller: Reset overrides on current frame",     icon: "⟲", keywords: ["face","reset","clear","undo"],     run: () => _fc3dActive()?.resetFrame()  },
        { id: "mec.faceController.gotoFirst",  title: "Face Controller: Go to first frame",                    icon: "⏮", keywords: ["face","first","home","frame"],      run: () => _fc3dActive()?.gotoFirst()   },
        { id: "mec.faceController.gotoLast",   title: "Face Controller: Go to last frame",                     icon: "⏭", keywords: ["face","last","end","frame"],        run: () => _fc3dActive()?.gotoLast()    },
        { id: "mec.faceController.viewFace",   title: "Face Controller: View — face only",                     icon: "👤", keywords: ["face","view","layout"],            run: () => _fc3dActive()?.setView("face") },
        { id: "mec.faceController.viewPose",   title: "Face Controller: View — pose only",                     icon: "🦴", keywords: ["face","pose","view","layout"],    run: () => _fc3dActive()?.setView("pose") },
        { id: "mec.faceController.viewBoth",   title: "Face Controller: View — face + pose",                   icon: "▦", keywords: ["face","pose","both","view","layout"], run: () => _fc3dActive()?.setView("both") },
        { id: "mec.faceController.mirrorRefresh", title: "Face Controller: Refresh live mirror (local)",        icon: "↻", keywords: ["face","live","mirror","preview"],   run: () => _fc3dActive()?.mirrorRefresh() },
        { id: "mec.faceController.serverResync",  title: "Face Controller: Re-sync from server now",            icon: "⇅", keywords: ["face","sync","server","resync","preview"], run: () => _fc3dActive()?.serverResync() },
        { id: "mec.faceController.resetTransform", title: "Face Controller: Reset all head/jaw/neck/gaze DOFs", icon: "⌖", keywords: ["face","reset","transform","head","jaw","neck","gaze","dof"], run: () => _fc3dActive()?.resetTransform() },
    ];
    for (const a of actions) {
        try { reg({ ...a, kind: "command", scope: "graph", enabled }); } catch (_) {}
    }
}
setTimeout(_fc3dRegisterActions, 0);
setTimeout(_fc3dRegisterActions, 1000);
