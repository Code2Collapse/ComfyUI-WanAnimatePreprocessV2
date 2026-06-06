// face_controller_3d.js — Tab-based editor for WanFaceController3DV2 (layout-v3)
//
// Complete rewrite: 4-tab editor (Face / Expr / Gaze / Pose) with a
// single main canvas, compact transport bar, and mini timeline.
// Backend contract (overlay_meta, widget JSON) is unchanged.

import { app } from "../../scripts/app.js";
/** Local labels only — avoids hard dependency on _c2c_i18n.js (load failure → blank widget). */
function T(_key, fallback) { return fallback ?? _key; }

const PROPAGATE_OPTS = ["off", "hold_last", "interpolate", "broadcast_first"];

const NODE_CLASS = "WanFaceController3DV2";
const OVERRIDE_WIDGET = "landmark_overrides_json";
const POSE_OVERRIDE_WIDGET = "pose_overrides_json";
const GAZE_OVERRIDE_WIDGET = "gaze_overrides_json";
const CONFIG_WIDGET = "fc3d_config_json";

/** Mirrors Python _FC3D_DEFAULT_CFG — editor-owned tunables blob. */
const FC3D_DEFAULT_CFG = {
    expression_coeffs_json: "",
    expression_strength: 1.0,
    expression_clamp: 1.5,
    propagate_expression: "off",
    head_pose_json: "",
    head_yaw_deg: 0.0,
    head_pitch_deg: 0.0,
    head_roll_deg: 0.0,
    head_tx: 0.0,
    head_ty: 0.0,
    head_tz: 0.0,
    head_scale: 1.0,
    jaw_rot_deg: 0.0,
    neck_yaw_deg: 0.0,
    neck_pitch_deg: 0.0,
    propagate_head: "off",
    propagate_gaze: "off",
    gaze_json: "",
    gaze_yaw_deg: 0.0,
    gaze_pitch_deg: 0.0,
    blend_strength: 0.0,
    blend_mouth: true,
    blend_brows: false,
    blend_eyes: false,
    blend_jaw: false,
    use_metas: "edited",
    frame_start: -1,
    frame_end: -1,
    preview_frame_idx: 0,
    preview_size: 512,
    preview_max_video_frames: 120,
};

const FC3D_PARAM_OPTS = {
    expression_strength: { min: -3, max: 3, step: 0.05 },
    expression_clamp: { min: 0.1, max: 3, step: 0.05 },
    head_yaw_deg: { min: -90, max: 90, step: 0.5 },
    head_pitch_deg: { min: -60, max: 60, step: 0.5 },
    head_roll_deg: { min: -60, max: 60, step: 0.5 },
    head_tx: { min: -2, max: 2, step: 0.01 },
    head_ty: { min: -2, max: 2, step: 0.01 },
    head_tz: { min: -0.75, max: 3, step: 0.01 },
    head_scale: { min: 0.25, max: 4, step: 0.01 },
    jaw_rot_deg: { min: -25, max: 25, step: 0.25 },
    neck_yaw_deg: { min: -60, max: 60, step: 0.5 },
    neck_pitch_deg: { min: -45, max: 45, step: 0.5 },
    gaze_yaw_deg: { min: -30, max: 30, step: 0.5 },
    gaze_pitch_deg: { min: -30, max: 30, step: 0.5 },
    blend_strength: { min: 0, max: 1, step: 0.05 },
    frame_start: { min: -1, max: 999999, step: 1 },
    frame_end: { min: -1, max: 999999, step: 1 },
    preview_frame_idx: { min: 0, max: 999999, step: 1 },
    preview_size: { min: 128, max: 2048, step: 32 },
    preview_max_video_frames: { min: 1, max: 1024, step: 1 },
};

// ─── Theme palette (CSS-var backed; hex fallbacks) ──────────────────
const _C_FALLBACK = {
    bg:     "#1a1a22", border: "#2a2a35", grid:   "#26263a",
    dim:    "#6c7086", sel:    "#cba6f7", emph:   "#fab387",
    other:  "#45475a", text:   "#cdd6f4", accent: "#89b4fa",
    pose_joint:"#a6e3a1", pose_bone:"#74c7ec", pose_selected:"#f9e2af",
    pose_missing:"#585b70",
    gaze_l:"#f38ba8", gaze_r:"#94e2d5", gaze_drag:"#f9e2af",
    canvas_bg:"#0e0e16", input_bg:"#1a1a23", btn_off_bg:"#22222e",
    fg_inverse:"#11111a", ok_bg:"#2d3b22", err_bg:"#3b2222", info_bg:"#222b3b",
    tab_active:"#313145",
};
const _C_TOKEN = {
    bg:"--c2c-bg", border:"--c2c-surface2", grid:"--c2c-surface1",
    dim:"--c2c-sub", sel:"--c2c-violetSoft", emph:"--c2c-yellow",
    other:"--c2c-surface2", text:"--c2c-fg", accent:"--c2c-blue",
    pose_joint:"--c2c-green", pose_bone:"--c2c-blue",
    pose_selected:"--c2c-yellow", pose_missing:"--c2c-sub",
    gaze_l:"--c2c-red", gaze_r:"--c2c-green", gaze_drag:"--c2c-yellow",
    canvas_bg:"--c2c-bg3", input_bg:"--c2c-bg2", btn_off_bg:"--c2c-surface0",
    fg_inverse:"--c2c-bg3", ok_bg:"--c2c-okBg", err_bg:"--c2c-dangerBg",
    info_bg:"--c2c-blueDim", tab_active:"--c2c-surface1",
};
const C = new Proxy(_C_FALLBACK, {
    get(target, key) {
        const tok = _C_TOKEN[key];
        if (tok) { try { const v = getComputedStyle(document.documentElement).getPropertyValue(tok).trim(); if (v) return v; } catch {} }
        return target[key];
    },
});

// ─── Data constants ─────────────────────────────────────────────────
const POSE18_NAMES = [
    "nose","neck","rShoulder","rElbow","rWrist","lShoulder","lElbow","lWrist",
    "rHip","rKnee","rAnkle","lHip","lKnee","lAnkle","rEye","lEye","rEar","lEar",
];
const POSE18_EDGES_DEFAULT = [
    [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
    [1,0],[0,14],[14,16],[0,15],[15,17],
];
const LM_NAMES = (() => {
    const out = new Array(68);
    for (let i=0;i<=16;i++) out[i]=`jaw_${i}`;
    for (let i=17;i<=21;i++) out[i]=`brow_r_${i-17}`;
    for (let i=22;i<=26;i++) out[i]=`brow_l_${i-22}`;
    for (let i=27;i<=30;i++) out[i]=`nose_bridge_${i-27}`;
    for (let i=31;i<=35;i++) out[i]=`nose_lower_${i-31}`;
    for (let i=36;i<=41;i++) out[i]=`eye_r_${i-36}`;
    for (let i=42;i<=47;i++) out[i]=`eye_l_${i-42}`;
    for (let i=48;i<=59;i++) out[i]=`mouth_outer_${i-48}`;
    for (let i=60;i<=67;i++) out[i]=`mouth_inner_${i-60}`;
    return out;
})();
const SEGMENTS = [
    [0,17],[17,22],[22,27],[27,31],[31,36],
    [36,42,true],[42,48,true],[48,60,true],[60,68,true],
];
const CANONICAL = [
    [0.05,0.45],[0.06,0.55],[0.08,0.65],[0.11,0.74],[0.16,0.82],
    [0.23,0.88],[0.32,0.93],[0.41,0.96],[0.50,0.97],[0.59,0.96],
    [0.68,0.93],[0.77,0.88],[0.84,0.82],[0.89,0.74],[0.92,0.65],
    [0.94,0.55],[0.95,0.45],
    [0.15,0.30],[0.22,0.26],[0.30,0.25],[0.38,0.26],[0.45,0.30],
    [0.55,0.30],[0.62,0.26],[0.70,0.25],[0.78,0.26],[0.85,0.30],
    [0.50,0.36],[0.50,0.43],[0.50,0.49],[0.50,0.55],
    [0.43,0.60],[0.46,0.62],[0.50,0.63],[0.54,0.62],[0.57,0.60],
    [0.22,0.40],[0.27,0.37],[0.33,0.37],[0.38,0.40],[0.33,0.43],[0.27,0.43],
    [0.62,0.40],[0.67,0.37],[0.73,0.37],[0.78,0.40],[0.73,0.43],[0.67,0.43],
    [0.34,0.74],[0.40,0.71],[0.46,0.70],[0.50,0.71],[0.54,0.70],[0.60,0.71],
    [0.66,0.74],[0.60,0.79],[0.54,0.81],[0.50,0.82],[0.46,0.81],[0.40,0.79],
    [0.38,0.74],[0.45,0.74],[0.50,0.74],[0.55,0.74],[0.62,0.74],
    [0.55,0.76],[0.50,0.77],[0.45,0.76],
];
const FACS_AXES = [
    {id:"brow_inner_raise",label:"Inner Brow Raise",au:"AU1",icon:"\u2934"},
    {id:"brow_outer_raise",label:"Outer Brow Raise",au:"AU2",icon:"\u2934"},
    {id:"brow_furrow",label:"Brow Furrow",au:"AU4",icon:"\u2322"},
    {id:"eye_close_L",label:"Eye Close L",au:"AU45L",icon:"\u25E0"},
    {id:"eye_close_R",label:"Eye Close R",au:"AU45R",icon:"\u25E0"},
    {id:"nose_wrinkle",label:"Nose Wrinkle",au:"AU9",icon:"\u224B"},
    {id:"cheek_raise",label:"Cheek Raise",au:"AU6",icon:"\u25E1"},
    {id:"smile",label:"Smile",au:"AU12",icon:"\u2323"},
    {id:"frown",label:"Frown",au:"AU15",icon:"\u2322"},
    {id:"mouth_open",label:"Mouth Open",au:"AU25",icon:"\u25CB"},
    {id:"jaw_drop",label:"Jaw Drop",au:"AU26",icon:"\u25BD"},
    {id:"lip_pucker",label:"Lip Pucker",au:"AU18",icon:"\u25CE"},
];
const EMOTION_PRESETS = {
    "Neutral":{},
    "Happy":{smile:0.8,cheek_raise:0.6,eye_close_L:0.15,eye_close_R:0.15},
    "Sad":{frown:0.7,brow_inner_raise:0.5,brow_furrow:0.3},
    "Angry":{brow_furrow:0.9,nose_wrinkle:0.5,lip_pucker:0.3,frown:0.4},
    "Surprise":{brow_inner_raise:0.8,brow_outer_raise:0.7,mouth_open:0.8,jaw_drop:0.5},
    "Fear":{brow_inner_raise:0.9,brow_outer_raise:0.4,mouth_open:0.5,eye_close_L:-0.2,eye_close_R:-0.2},
    "Disgust":{nose_wrinkle:0.8,brow_furrow:0.4,frown:0.5,cheek_raise:0.3,lip_pucker:0.4},
    "Wink L":{eye_close_L:1.0,smile:0.4,cheek_raise:0.3},
    "Wink R":{eye_close_R:1.0,smile:0.4,cheek_raise:0.3},
};
const PRESET_QUICK = ["Neutral", "Happy", "Sad", "Angry", "Surprise"];
const PRESET_MORE = ["Fear", "Disgust", "Wink L", "Wink R"];

/** Face-bbox-normalised FACS deltas (must match expression_3d_coeffs._BASIS_TABLE). */
const FACS_BASIS = {
    brow_inner_raise: {20:[0,-0.04],21:[0,-0.08],22:[0,-0.08],23:[0,-0.04]},
    brow_outer_raise: {17:[0,-0.07],18:[0,-0.05],25:[0,-0.05],26:[0,-0.07]},
    brow_furrow: {17:[0.005,0.015],18:[0.01,0.02],19:[0.015,0.025],20:[0.02,0.03],21:[0.025,0.035],22:[-0.025,0.035],23:[-0.02,0.03],24:[-0.015,0.025],25:[-0.01,0.02],26:[-0.005,0.015]},
    eye_close_L: {43:[0,0.025],44:[0,0.025],46:[0,-0.005],47:[0,-0.005]},
    eye_close_R: {37:[0,0.025],38:[0,0.025],40:[0,-0.005],41:[0,-0.005]},
    nose_wrinkle: {30:[0,-0.015],33:[0,-0.02],31:[0,-0.02],35:[0,-0.02]},
    cheek_raise: {40:[0,-0.01],41:[0,-0.01],46:[0,-0.01],47:[0,-0.01],48:[-0.003,-0.005],54:[0.003,-0.005]},
    smile: {48:[-0.025,-0.03],49:[-0.015,-0.02],50:[-0.005,-0.01],52:[0.005,-0.01],53:[0.015,-0.02],54:[0.025,-0.03],60:[-0.015,-0.018],64:[0.015,-0.018]},
    frown: {48:[-0.005,0.03],49:[-0.002,0.02],50:[0,0.01],52:[0,0.01],53:[0.002,0.02],54:[0.005,0.03]},
    mouth_open: {56:[0,0.02],57:[0,0.025],58:[0,0.02],65:[0,0.02],66:[0,0.025],67:[0,0.02],50:[0,-0.004],51:[0,-0.006],52:[0,-0.004],61:[0,-0.004],62:[0,-0.006],63:[0,-0.004]},
    jaw_drop: {6:[0,0.035],7:[0,0.05],8:[0,0.06],9:[0,0.05],10:[0,0.035],56:[0,0.03],57:[0,0.04],58:[0,0.03],65:[0,0.03],66:[0,0.04],67:[0,0.03]},
    lip_pucker: {48:[0.02,0],54:[-0.02,0],49:[0.01,0],53:[-0.01,0],51:[0,-0.005],57:[0,0.005],62:[0,-0.003],66:[0,0.003]},
};

const CANVAS_MIN_PX = 200;
const CANVAS_MAX_PX = 420;
const FC3D_CANVAS_VIEW_PX = 260;
const FC3D_NODE_W = 300;
const FC3D_MIN_H = 340;
const FC3D_MAX_H = 680;
const FC3D_CTX_SCROLL_MAX = 220;
/** Per-tab canvas height caps — keeps total node under ~650px with context panels. */
const FC3D_TAB_CANVAS_PX = { face: 260, expr: 240, gaze: 220, pose: 260, set: 200 };
const FC3D_TAB_CTX_MAX = { face: 80, expr: 200, gaze: 110, pose: 160, set: 200 };

const FC3D_HIDE_STYLE_ID = "fc3d-wan-face-hide-css";

/** Collapse chrome widgets in both LiteGraph canvas mode AND Vue mode. */
function _fc3dHideChromeWidget(w) {
    if (!w || w.name === "face_overlay") return;
    w.__fc3d_chrome_hidden = true;
    if (w.options === undefined) w.options = {};

    // Vue frontend checks widget.options.hidden to skip rendering entirely
    w.options.hidden = true;

    // LiteGraph canvas mode + Vue layout
    if (w.__fc3d_origType === undefined) w.__fc3d_origType = w.type;
    if (w.__fc3d_origComputeSize === undefined) w.__fc3d_origComputeSize = w.computeSize;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    w.draw = () => {};
    if (w.options.getMinHeight === undefined) w.options.getMinHeight = () => 0;
    if (w.options.getHeight === undefined) w.options.getHeight = () => 0;
    if (w.options.getMaxHeight === undefined) w.options.getMaxHeight = () => 0;

    // DOM elements (multiline STRING textareas)
    const el = w.element ?? w.inputEl;
    if (el) {
        el.hidden = true;
        el.style.display = "none";
        el.style.height = "0";
        el.style.overflow = "hidden";
    }
    // Hide the wrapper container too
    const wrap = el?.parentElement;
    if (wrap && (wrap.classList?.contains("dom-widget") ||
                 wrap.classList?.contains("lg-node-widget") ||
                 wrap.classList?.contains("comfy-widget"))) {
        wrap.style.display = "none";
        wrap.style.height = "0";
        wrap.style.overflow = "hidden";
    }
}

function _fc3dEnsureHideCss() {
    if (document.getElementById(FC3D_HIDE_STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = FC3D_HIDE_STYLE_ID;
    st.textContent = `
.fc3d-wan-face .comfy-widget:not(:has(.fc3d-editor-root)),
.fc3d-wan-face .lg-node-widget:not(:has(.fc3d-editor-root)):not([class*="resize"]) {
    display: none !important; height: 0 !important; max-height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important; padding: 0 !important; overflow: hidden !important;
    position: absolute !important; pointer-events: none !important;
}
`;
    document.head.appendChild(st);
}

function _fc3dHideChromeWidgets(node, domW) {
    _fc3dEnsureHideCss();
    // Apply the class to the Vue node container for CSS rules
    if (node.el) node.el.classList.add("fc3d-wan-face");
    // Also try to find the node container from the DOM widget's element
    const editorEl = domW?.element;
    if (editorEl) {
        const nodeContainer = editorEl.closest("[data-node-id]") || editorEl.closest(".graphnode") || editorEl.closest(".node");
        if (nodeContainer) nodeContainer.classList.add("fc3d-wan-face");
    }
    for (const wg of node.widgets || []) {
        if (wg !== domW) _fc3dHideChromeWidget(wg);
    }
}

/** LiteGraph node.size is array-like but not always Array.isArray — mutate indices directly. */
function _fc3dWriteNodeSize(node, w, h) {
    if (!node) return;
    if (!node.size) node.size = [w, h];
    else {
        node.size[0] = w;
        node.size[1] = h;
    }
}

/**
 * Simplified size management — follows official ComfyUI DOMWidget API.
 * No setSize override, no clamp loops, no snap-back.
 */
function _fc3dGetContentHeight(node) {
    return Math.min(FC3D_MAX_H, Math.max(FC3D_MIN_H, node._faceOverlay?.getHeight?.() || FC3D_MIN_H));
}

function _fc3dSetupDomWidget(node, domW) {
    const contentH = () => _fc3dGetContentHeight(node);

    domW.computeSize = (width) => [
        Math.max(FC3D_NODE_W, width || node.size?.[0] || FC3D_NODE_W),
        contentH(),
    ];
    domW.computeLayoutSize = () => ({
        minHeight: FC3D_MIN_H,
        maxHeight: FC3D_MAX_H,
    });
    if (!domW.options) domW.options = {};
    domW.options.getMinHeight = () => FC3D_MIN_H;
    domW.options.getHeight = contentH;
    domW.options.getMaxHeight = () => FC3D_MAX_H;
    domW.options.afterResize = function () {
        try { node._faceOverlay?._relayout?.(); } catch (_) {}
        try { node._faceOverlay?.render?.(); node._faceOverlay?.drawTimeline?.(); } catch (_) {}
    };
    if (domW.element) {
        domW.element.style.overflow = "hidden";
        domW.element.style.width = "100%";
        domW.element.style.boxSizing = "border-box";
        domW.element.style.background = C.bg;
        domW.element.style.padding = "0";
    }
}

function _fc3dSyncNodeSize(node) {
    if (!node) return;
    try {
        const sz = node.computeSize();
        node.setSize(sz);
        node.setDirtyCanvas?.(true, true);
    } catch (_) {}
}

function makeDefaultMeta() {
    const zeros = Array.from({length:68},()=>[0,0]);
    const selected = Array.from({length:68},(_,i)=>i);
    return { selected, eye_emph:[37,38,43,44], d_norm:zeros,
             frames:[{i:0,ok:true}], strength:1.0,
             pose:{format:"openpose_18",joint_names:POSE18_NAMES,edges:POSE18_EDGES_DEFAULT,frames:[]},
             gaze:null, _synthetic:true };
}

// ─── Widget + Override helpers ──────────────────────────────────────
function readWidget(node,name) { return node.widgets?.find(w=>w.name===name); }

function _parseFc3dConfig(node) {
    const w = readWidget(node, CONFIG_WIDGET);
    if (!w?.value) return { ...FC3D_DEFAULT_CFG };
    try {
        const o = JSON.parse(w.value);
        return (o && typeof o === "object") ? { ...FC3D_DEFAULT_CFG, ...o } : { ...FC3D_DEFAULT_CFG };
    } catch (_) {
        return { ...FC3D_DEFAULT_CFG };
    }
}

function _writeFc3dConfig(node, cfg) {
    const w = readWidget(node, CONFIG_WIDGET);
    if (!w) return;
    const s = JSON.stringify(cfg);
    if (w.value === s) return;
    w.value = s;
    try { w.callback?.(s, node, w); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
}

function readParam(node, key) {
    const cfg = _parseFc3dConfig(node);
    if (Object.prototype.hasOwnProperty.call(cfg, key)) return cfg[key];
    const w = readWidget(node, key);
    if (w) return w.value;
    return FC3D_DEFAULT_CFG[key];
}

function writeParam(node, key, val) {
    const cfg = _parseFc3dConfig(node);
    if (cfg[key] === val) return;
    cfg[key] = val;
    _writeFc3dConfig(node, cfg);
    const legacy = readWidget(node, key);
    if (legacy && legacy.value !== val) {
        legacy.value = val;
        try { legacy.callback?.(val, node, legacy); } catch (_) {}
    }
    node.setDirtyCanvas?.(true, true);
}

function _fc3dMigrateLegacyWidgets(node) {
    const cfg = _parseFc3dConfig(node);
    let changed = false;
    for (const key of Object.keys(FC3D_DEFAULT_CFG)) {
        const lw = readWidget(node, key);
        if (!lw || lw === readWidget(node, CONFIG_WIDGET)) continue;
        const cur = cfg[key];
        const def = FC3D_DEFAULT_CFG[key];
        const lv = lw.value;
        if (lv === undefined || lv === null || lv === "") continue;
        if (cur === def && lv !== def && String(lv) !== String(def)) {
            cfg[key] = lv;
            changed = true;
        }
    }
    if (changed) _writeFc3dConfig(node, cfg);
}

function _fc3dEmitParam(node, key, val, hooks) {
    writeParam(node, key, val);
    try { hooks?.(); } catch (_) {}
}

const _fc3dPending = new WeakMap();
function _fc3dSchedulePersist(node) {
    let st = _fc3dPending.get(node);
    if (!st) { st={ov:null,pose:null,gaze:null,raf:0}; _fc3dPending.set(node,st); }
    if (st.raf) return st;
    st.raf = requestAnimationFrame(()=>{
        st.raf=0;
        const flush=(key,wn)=>{
            const data=st[key]; if(!data) return; st[key]=null;
            const w=readWidget(node,wn); if(!w) return;
            w.value=JSON.stringify(data,null,0);
            if(w.callback) try{w.callback(w.value);}catch(_){}
        };
        flush("ov",OVERRIDE_WIDGET); flush("pose",POSE_OVERRIDE_WIDGET); flush("gaze",GAZE_OVERRIDE_WIDGET);
        node.setDirtyCanvas?.(true,true);
        try{node._faceOverlay?._scheduleUndoSnap?.();}catch(_){}
    });
    return st;
}

function parseOverrides(node) {
    const p=_fc3dPending.get(node); if(p&&p.ov) return p.ov;
    const w=readWidget(node,OVERRIDE_WIDGET);
    if(!w||!w.value) return {frames:{}};
    try{const o=JSON.parse(w.value);if(!o.frames||typeof o.frames!=="object")o.frames={};return o;}catch(_){return {frames:{}};}
}
function writeOverrides(node,ov) { if(!readWidget(node,OVERRIDE_WIDGET))return; _fc3dSchedulePersist(node).ov=ov; }

function parsePoseOverrides(node) {
    const p=_fc3dPending.get(node); if(p&&p.pose) return p.pose;
    const w=readWidget(node,POSE_OVERRIDE_WIDGET);
    if(!w||!w.value) return {frames:{}};
    try{const o=JSON.parse(w.value);if(!o.frames||typeof o.frames!=="object")o.frames={};return o;}catch(_){return {frames:{}};}
}
function writePoseOverrides(node,ov) { if(!readWidget(node,POSE_OVERRIDE_WIDGET))return; _fc3dSchedulePersist(node).pose=ov; }

function parseGazeOverrides(node) {
    const p=_fc3dPending.get(node); if(p&&p.gaze) return p.gaze;
    const w=readWidget(node,GAZE_OVERRIDE_WIDGET);
    if(!w||!w.value) return {frames:{}};
    try{const o=JSON.parse(w.value);if(!o.frames||typeof o.frames!=="object")o.frames={};return o;}catch(_){return {frames:{}};}
}
function writeGazeOverrides(node,ov) { if(!readWidget(node,GAZE_OVERRIDE_WIDGET))return; _fc3dSchedulePersist(node).gaze=ov; }

function denormToCanvas(xn,yn,W,H) { return [xn*W,yn*H]; }
function canvasToNorm(cx,cy,W,H) { return [Math.max(0,Math.min(1,cx/W)),Math.max(0,Math.min(1,cy/H))]; }

// ─── DOM factory helpers (inspired by enricos-nodes pattern) ────────
function _el(tag, style, attrs) {
    const e = document.createElement(tag);
    if (style) e.style.cssText = style;
    if (attrs) for (const [k,v] of Object.entries(attrs)) { if (k === "text") e.textContent = v; else e.setAttribute(k,v); }
    return e;
}
function _btn(label, title, bg) {
    const b = _el("button",
        `background:${bg||C.border};color:${C.text};border:1px solid ${C.border};` +
        `border-radius:4px;padding:3px 8px;cursor:pointer;font:11px ui-sans-serif,system-ui;line-height:1;`);
    b.textContent = label; b.title = title || "";
    b.onmousedown = (e) => e.stopPropagation();
    return b;
}

/** Sync in-tab control → hidden Comfy widget (backend still reads widgets). */
function _fc3dEmitWidget(node, w, val, hooks) {
    if (!w || w.value === val) return;
    w.value = val;
    try { w.callback?.(val, node, w); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
    try { hooks?.(); } catch (_) {}
}

// ═══════════════════════════════════════════════════════════════════
//  buildEditor — tabbed editor factory (replaces old buildOverlay)
// ═══════════════════════════════════════════════════════════════════
function buildEditor(node) {
    if (node._faceOverlay) return node._faceOverlay;

    let activeTab = "face";
    let _onWidgetChg = () => {};
    const COEFF_WIDGET = "expression_coeffs_json";

    // ── Root container (inset so LiteGraph resize grip / border stay clickable) ──
    const root = _el("div",
        `display:flex;flex-direction:column;gap:4px;box-sizing:border-box;` +
        `width:calc(100% - 12px);margin:2px 6px 4px 6px;` +
        `height:auto;max-height:100%;align-self:stretch;` +
        `background:${C.bg};border:1px solid ${C.border};border-radius:6px;` +
        `padding:6px;overflow:hidden;font:12px ui-sans-serif,system-ui;color:${C.text};` +
        `user-select:none;pointer-events:auto;`);
    root.classList.add("fc3d-editor-root");
    root.style.flex = "1 1 auto";

    // Forward middle-mouse to canvas so workflow panning works over the editor
    root.addEventListener("pointerdown", (e) => {
        if (e.button === 1) { try { app.canvas?.processMouseDown(e); } catch (_) {} }
    });
    root.addEventListener("pointermove", (e) => {
        if ((e.buttons & 4) === 4) { try { app.canvas?.processMouseMove(e); } catch (_) {} }
    });
    root.addEventListener("pointerup", (e) => {
        if (e.button === 1) { try { app.canvas?.processMouseUp(e); } catch (_) {} }
    });

    // ── Tab bar ──────────────────────────────────────────────────────
    const tabBar = _el("div",
        `display:flex;gap:2px;background:${C.canvas_bg};border:1px solid ${C.border};` +
        `border-radius:6px;padding:2px;overflow:hidden;pointer-events:auto;flex:0 0 auto;`);
    const frameHint = _el("span",
        `font:9px ui-monospace,monospace;color:${C.dim};padding:0 4px;flex:0 0 auto;align-self:center;`);
    frameHint.textContent = "";

    const TABS = [
        {id:"face", label:"Face"},
        {id:"expr", label:"Expr"},
        {id:"gaze", label:"Gaze"},
        {id:"pose", label:"Pose"},
        {id:"set", label:"Settings"},
    ];
    const tabBtns = {};
    for (const t of TABS) {
        const b = _el("button",
            `flex:1;padding:6px 4px;border:none;border-radius:4px;cursor:pointer;` +
            `font:11px ui-sans-serif,system-ui;font-weight:600;` +
            `background:${t.id==="face"?C.tab_active:"transparent"};color:${C.text};`);
        b.textContent = t.label;
        b.onmousedown = (e) => e.stopPropagation();
        b.addEventListener("click", () => switchTab(t.id));
        tabBar.appendChild(b);
        tabBtns[t.id] = b;
    }
    tabBar.appendChild(frameHint);
    root.appendChild(tabBar);

    // ── Main canvas (fixed height — flex:1 was absorbing ComfyUI grid slack → black slab) ──
    let _canvasViewPx = FC3D_CANVAS_VIEW_PX;
    const canvasWrap = _el("div",
        `position:relative;flex:0 0 auto;width:100%;overflow:hidden;background:${C.canvas_bg};` +
        `border:1px solid ${C.border};border-radius:6px;pointer-events:auto;cursor:crosshair;`);
    const cvs = _el("canvas", "display:block;width:100%;height:100%;cursor:crosshair;outline:none;");
    let _canvasPx = FC3D_CANVAS_VIEW_PX;
    function _applyCanvasViewPx(px) {
        const side = Math.max(CANVAS_MIN_PX, Math.min(CANVAS_MAX_PX, Math.round(px)));
        _canvasViewPx = side;
        canvasWrap.style.height = `${side}px`;
        canvasWrap.style.minHeight = `${side}px`;
        canvasWrap.style.maxHeight = `${side}px`;
        if (side === _canvasPx && cvs.width === side) return false;
        _canvasPx = side;
        cvs.width = side;
        cvs.height = side;
        return true;
    }
    function _syncCanvasPx() {
        const w = Math.max(CANVAS_MIN_PX, Math.min(CANVAS_MAX_PX, Math.floor(canvasWrap.clientWidth || _canvasViewPx)));
        return _applyCanvasViewPx(w);
    }
    _applyCanvasViewPx(FC3D_CANVAS_VIEW_PX);
    cvs.tabIndex = 0;
    cvs.title = "drag to edit \u00b7 arrow keys step frame \u00b7 R reset frame";
    /** Undo LiteGraph zoom scale so drags land on the correct landmark. */
    function eventCanvas(e) {
        const r = cvs.getBoundingClientRect();
        const sx = (cvs.offsetWidth || r.width) / Math.max(1, r.width);
        const sy = (cvs.offsetHeight || r.height) / Math.max(1, r.height);
        return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
    }
    cvs.addEventListener("focus", () => { cvs.style.boxShadow = `0 0 0 2px ${C.accent}`; });
    cvs.addEventListener("blur",  () => { cvs.style.boxShadow = "none"; });
    canvasWrap.appendChild(cvs);
    root.appendChild(canvasWrap);

    // ── Transport bar ───────────────────────────────────────────────
    const transport = _el("div",
        `display:flex;align-items:center;gap:4px;padding:4px 6px;flex:0 0 auto;pointer-events:auto;` +
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;`);
    const btnPrev = _btn("\u25C0","Previous frame"); btnPrev.style.padding="2px 6px";
    const frameLbl = _el("span",`font:10px ui-monospace,monospace;color:${C.dim};min-width:64px;text-align:center;`);
    frameLbl.textContent = "f 0 / 0";
    const slider = _el("input",`flex:1;height:14px;accent-color:${C.sel};cursor:pointer;min-width:60px;`);
    slider.type="range"; slider.min="0"; slider.max="0"; slider.value="0";
    const btnNext = _btn("\u25B6","Next frame"); btnNext.style.padding="2px 6px";
    const btnUndo = _btn("\u21B6","Undo (Ctrl+Z)"); btnUndo.disabled=true;
    const btnRedo = _btn("\u21B7","Redo (Ctrl+Shift+Z)"); btnRedo.disabled=true;
    const btnReset = _btn("\u21BA","Reset this frame",C.err_bg);
    transport.append(btnPrev,frameLbl,slider,btnNext,btnUndo,btnRedo,btnReset);
    root.appendChild(transport);

    // ── Mini timeline ───────────────────────────────────────────────
    const tl = _el("canvas",
        `width:100%;height:20px;display:block;background:${C.canvas_bg};flex:0 0 auto;pointer-events:auto;` +
        `border:1px solid ${C.border};border-radius:4px;cursor:pointer;`);
    tl.width = 480; tl.height = 20;
    tl.title = "click=jump \u00b7 shift-click=range \u00b7 right-click=clear frame";
    root.appendChild(tl);

    // ── Context panels (one per tab, shown/hidden) ──────────────────
    const ctxPanels = {};
    const _ctxPanelStyle = "flex:0 0 auto;pointer-events:auto;";

    // -- Face context: collapsed numeric editor
    const ctxFace = _el("div", _ctxPanelStyle);
    const faceDetails = _el("details",
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:6px;overflow:hidden;`);
    const faceSummary = _el("summary",
        `padding:6px 10px;cursor:pointer;font:11px ui-sans-serif;color:${C.dim};user-select:none;`);
    faceSummary.textContent = "\u25B8 Numeric coordinates";
    faceDetails.appendChild(faceSummary);
    faceDetails.addEventListener("toggle",()=>{
        faceSummary.textContent = (faceDetails.open?"\u25BE ":"\u25B8 ")+"Numeric coordinates";
        _relayout();
    });
    const faceEdRow = _el("div",
        `display:flex;flex-wrap:wrap;align-items:center;gap:4px;padding:6px 8px;font:10px ui-sans-serif;color:${C.text};`);
    const _mkNumIn = (ph,step,w) => {
        const i=_el("input",`width:${w}px;padding:3px 4px;background:${C.input_bg};color:${C.text};` +
            `border:1px solid ${C.border};border-radius:3px;font:10px ui-monospace,monospace;outline:none;`);
        i.type="number";i.step=String(step);i.placeholder=ph; return i;
    };
    const tgtSel = _el("select",`padding:3px;background:${C.input_bg};color:${C.text};border:1px solid ${C.border};border-radius:3px;font:10px ui-sans-serif;cursor:pointer;outline:none;`);
    for(const[v,t]of[["face","face"],["pose","pose"],["gaze-l","gaze L"],["gaze-r","gaze R"]]){const o=_el("option");o.value=v;o.textContent=t;tgtSel.appendChild(o);}
    const idxIn=_mkNumIn("idx",1,36); idxIn.value="30"; idxIn.min="0"; idxIn.max="67";
    const xIn=_mkNumIn("x",0.001,56);
    const yIn=_mkNumIn("y",0.001,56);
    const btnSet=_btn("Set","Write override",C.ok_bg); btnSet.style.fontSize="10px";btnSet.style.padding="2px 6px";
    const btnClear=_btn("Clear","Clear override",C.err_bg); btnClear.style.fontSize="10px";btnClear.style.padding="2px 6px";
    const nameTag=_el("span",`color:${C.dim};font:9px ui-monospace,monospace;margin-left:auto;`);
    nameTag.textContent="\u2014";
    faceEdRow.append(tgtSel,idxIn,xIn,yIn,btnSet,btnClear,nameTag);
    faceDetails.appendChild(faceEdRow);
    ctxFace.appendChild(faceDetails);
    ctxPanels.face = ctxFace;

    function _refreshNameTag() {
        const t=tgtSel.value;
        if(t==="face"){const i=parseInt(idxIn.value,10)||0;nameTag.textContent=`${LM_NAMES[i]||"?"}[${i}]`;}
        else if(t==="pose"){const i=parseInt(idxIn.value,10)||0;nameTag.textContent=`${POSE18_NAMES[i]||"?"}[${i}]`;}
        else nameTag.textContent=t==="gaze-l"?"gaze L":"gaze R";
    }
    tgtSel.addEventListener("change",()=>{
        const isGaze=tgtSel.value.startsWith("gaze");
        idxIn.style.display=isGaze?"none":"";
        idxIn.max=tgtSel.value==="pose"?"17":"67";
        _refreshNameTag();
    });
    idxIn.addEventListener("input",_refreshNameTag);
    _refreshNameTag();

    btnSet.addEventListener("click",()=>{
        const t=tgtSel.value,x=Number(xIn.value),y=Number(yIn.value),key=String(state.frame);
        if(!isFinite(x)||!isFinite(y))return;
        if(t==="face"){
            const i=parseInt(idxIn.value,10); if(!(i>=0&&i<=67))return;
            const ov=parseOverrides(node); if(!ov.frames[key])ov.frames[key]={};
            ov.frames[key][String(i)]=[+Math.max(0,Math.min(1,x)).toFixed(4),+Math.max(0,Math.min(1,y)).toFixed(4)];
            writeOverrides(node,ov);
        } else if(t==="pose"){
            const i=parseInt(idxIn.value,10); if(!(i>=0&&i<=17))return;
            const ov=parsePoseOverrides(node); if(!ov.frames[key])ov.frames[key]={};
            ov.frames[key][String(i)]=[+Math.max(0,Math.min(1,x)).toFixed(4),+Math.max(0,Math.min(1,y)).toFixed(4)];
            writePoseOverrides(node,ov);
        } else {
            const eye=t==="gaze-l"?"l":"r",cap=Math.PI/2;
            const ov=parseGazeOverrides(node); if(!ov.frames[key])ov.frames[key]={};
            ov.frames[key][eye]=[+Math.max(-cap,Math.min(cap,x)).toFixed(5),+Math.max(-cap,Math.min(cap,y)).toFixed(5)];
            writeGazeOverrides(node,ov);
        }
        render(); drawTimeline();
    });
    btnClear.addEventListener("click",()=>{
        const t=tgtSel.value,key=String(state.frame);
        if(t==="face"){const i=parseInt(idxIn.value,10),ov=parseOverrides(node);if(ov.frames?.[key]){delete ov.frames[key][String(i)];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writeOverrides(node,ov);}}
        else if(t==="pose"){const i=parseInt(idxIn.value,10),ov=parsePoseOverrides(node);if(ov.frames?.[key]){delete ov.frames[key][String(i)];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writePoseOverrides(node,ov);}}
        else{const eye=t==="gaze-l"?"l":"r",ov=parseGazeOverrides(node);if(ov.frames?.[key]){delete ov.frames[key][eye];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writeGazeOverrides(node,ov);}}
        render(); drawTimeline();
    });

    // -- Expression context: quick presets + collapsed fine-tune
    const ctxExpr = _el("div", _ctxPanelStyle + "display:flex;flex-direction:column;gap:3px;");
    const presetBar = _el("div","display:flex;flex-wrap:wrap;gap:2px;align-items:center;");
    let _activePreset = "Neutral";
    function _markPreset(name) {
        _activePreset = name;
        for (const ch of presetBar.children) {
            if (ch.tagName !== "BUTTON") continue;
            const on = ch.textContent === name;
            ch.style.background = on ? C.tab_active : C.btn_off_bg;
            ch.style.borderColor = on ? C.sel : C.border;
        }
    }
    for (const name of PRESET_QUICK) {
        const b = _el("button",
            `padding:2px 7px;font:10px ui-sans-serif;border-radius:4px;cursor:pointer;` +
            `background:${name==="Neutral"?C.tab_active:C.btn_off_bg};color:${C.text};border:1px solid ${C.border};`);
        b.textContent = name;
        b.addEventListener("click", () => { _applyPreset(name); _markPreset(name); });
        presetBar.appendChild(b);
    }
    const moreSel = _el("select",
        `font:10px ui-sans-serif;background:${C.input_bg};color:${C.dim};border:1px solid ${C.border};` +
        `border-radius:4px;padding:2px 4px;cursor:pointer;margin-left:auto;`);
    const moreOpt0 = _el("option"); moreOpt0.value = ""; moreOpt0.textContent = "More\u2026";
    moreSel.appendChild(moreOpt0);
    for (const name of PRESET_MORE) {
        const o = _el("option"); o.value = name; o.textContent = name;
        moreSel.appendChild(o);
    }
    moreSel.addEventListener("change", () => {
        if (!moreSel.value) return;
        _applyPreset(moreSel.value);
        _markPreset(moreSel.value);
        moreSel.value = "";
    });
    presetBar.appendChild(moreSel);
    const btnResetExpr = _btn("Clr", "Clear expression on this frame", C.err_bg);
    btnResetExpr.style.cssText = "font-size:10px;padding:2px 6px;margin-left:2px;";
    btnResetExpr.addEventListener("click", () => {
        const d = _parseCoeffsJson();
        delete d.frames[String(state.frame)];
        _writeCoeffsJson(d);
        _refreshSliders();
        _markPreset("Neutral");
        render();
    });
    presetBar.appendChild(btnResetExpr);
    ctxExpr.appendChild(presetBar);

    const exprFineTune = _el("details",
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;`);
    const exprFtSum = _el("summary",
        `padding:4px 8px;cursor:pointer;font:10px ui-sans-serif;color:${C.dim};user-select:none;`);
    exprFtSum.textContent = "\u25B8 Fine-tune";
    exprFineTune.appendChild(exprFtSum);

    const sliderPrimary = _el("div",
        `display:grid;grid-template-columns:1fr 1fr;gap:2px 6px;padding:4px 6px;`);
    const _sliderEls = {};

    function _addSliderRow(axis, parent) {
        const row = _el("div", `display:grid;grid-template-columns:52px 1fr 28px;align-items:center;gap:2px;height:18px;`);
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`);
        lbl.textContent = axis.label.split(" ").slice(-1)[0] || axis.id;
        lbl.title = axis.label;
        const sl = _el("input", `width:100%;height:10px;accent-color:${C.sel};cursor:pointer;`);
        sl.type = "range"; sl.min = "-1"; sl.max = "1"; sl.step = "0.01"; sl.value = "0";
        const val = _el("span", `font:9px ui-monospace,monospace;color:${C.dim};text-align:right;`);
        val.textContent = "0";
        sl.addEventListener("input", () => {
            const v = parseFloat(sl.value);
            val.textContent = v.toFixed(2);
            val.style.color = Math.abs(v) > 0.01 ? C.sel : C.dim;
            _setCoeff(state.frame, axis.id, v);
            if (activeTab === "expr") render();
        });
        sl.addEventListener("dblclick", () => {
            sl.value = "0"; val.textContent = "0.00"; val.style.color = C.dim;
            _setCoeff(state.frame, axis.id, 0);
            if (activeTab === "expr") render();
        });
        row.append(lbl, sl, val);
        parent.appendChild(row);
        _sliderEls[axis.id] = { sl, val };
    }

    for (const axis of FACS_AXES) _addSliderRow(axis, sliderPrimary);
    exprFineTune.appendChild(sliderPrimary);

    const exprPropRow = _el("div", "display:flex;align-items:center;gap:4px;padding:2px 6px 4px;flex-wrap:wrap;");
    const exprPropLbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};`);
    exprPropLbl.textContent = "Propagate";
    const exprPropSel = _el("select",
        `font:9px ui-sans-serif;flex:1;min-width:100px;padding:2px;background:${C.input_bg};color:${C.text};` +
        `border:1px solid ${C.border};border-radius:3px;`);
    for (const v of PROPAGATE_OPTS) {
        const o = _el("option"); o.value = v; o.textContent = v;
        exprPropSel.appendChild(o);
    }
    const wPropExpr = readWidget(node, "propagate_expression");
    exprPropSel.value = String(readParam(node, "propagate_expression") || "off");
    exprPropSel.addEventListener("change", () =>
        _fc3dEmitParam(node, "propagate_expression", exprPropSel.value, _onWidgetChg));
    exprPropRow.append(exprPropLbl, exprPropSel);
    exprFineTune.appendChild(exprPropRow);
    exprFineTune.open = false;

    exprFineTune.addEventListener("toggle", () => {
        exprFtSum.textContent = (exprFineTune.open ? "\u25BE " : "\u25B8 ") + "Fine-tune (12 axes)";
        _relayout();
    });

    ctxExpr.appendChild(exprFineTune);
    ctxPanels.expr = ctxExpr;

    // -- Gaze context: compact dual gimbal
    const ctxGaze = _el("div", _ctxPanelStyle + "display:flex;align-items:center;justify-content:center;gap:6px;padding:2px 0;");
    let gazeLinked = true;
    const GIMBAL_PX = 72;
    function _buildGimbal(label,eyeKey,color) {
        const wrap = _el("div","display:flex;flex-direction:column;align-items:center;gap:2px;");
        const eyeLbl = _el("span",`font:9px ui-sans-serif;font-weight:600;color:${color};`);eyeLbl.textContent=label;
        const gCvs = _el("canvas",`width:${GIMBAL_PX}px;height:${GIMBAL_PX}px;border-radius:50%;cursor:crosshair;background:${C.bg};border:1px solid ${C.border};`);
        gCvs.width=GIMBAL_PX;gCvs.height=GIMBAL_PX;
        const info = _el("span",`font:9px ui-monospace,monospace;color:${C.dim};`);info.textContent="0.0\u00b0 / 0.0\u00b0";
        const SZ=GIMBAL_PX,R=SZ/2-5,MAX_DEG=30;
        function drawG() {
            const ctx=gCvs.getContext("2d");ctx.clearRect(0,0,SZ,SZ);
            ctx.strokeStyle=C.border;ctx.lineWidth=1;
            for(let r2=1;r2<=3;r2++){ctx.beginPath();ctx.arc(SZ/2,SZ/2,(R/3)*r2,0,Math.PI*2);ctx.stroke();}
            ctx.beginPath();ctx.moveTo(SZ/2,4);ctx.lineTo(SZ/2,SZ-4);ctx.stroke();
            ctx.beginPath();ctx.moveTo(4,SZ/2);ctx.lineTo(SZ-4,SZ/2);ctx.stroke();
            const g=_getGazeForFrame(state.frame,eyeKey);
            const px=SZ/2+(g.yaw/MAX_DEG)*R, py=SZ/2-(g.pitch/MAX_DEG)*R;
            ctx.fillStyle=color;ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();
            ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(px,py,6,0,Math.PI*2);ctx.stroke();
            info.textContent=`${g.yaw.toFixed(1)}\u00b0 / ${g.pitch.toFixed(1)}\u00b0`;
        }
        let dragging=false;
        function m2g(e){const rect=gCvs.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;return{yaw:Math.max(-MAX_DEG,Math.min(MAX_DEG,((mx-SZ/2)/R)*MAX_DEG)),pitch:Math.max(-MAX_DEG,Math.min(MAX_DEG,((SZ/2-my)/R)*MAX_DEG))};}
        gCvs.addEventListener("mousedown",e=>{e.preventDefault();dragging=true;const g=m2g(e);_setGazeForFrame(state.frame,eyeKey,g.yaw,g.pitch);if(gazeLinked)_setGazeForFrame(state.frame,eyeKey==="l"?"r":"l",g.yaw,g.pitch);_drawGimbals();render();});
        gCvs.addEventListener("mousemove",e=>{if(!dragging)return;const g=m2g(e);_setGazeForFrame(state.frame,eyeKey,g.yaw,g.pitch);if(gazeLinked)_setGazeForFrame(state.frame,eyeKey==="l"?"r":"l",g.yaw,g.pitch);_drawGimbals();render();});
        const up=()=>{dragging=false;};
        window.addEventListener("mouseup",up);
        if(!root._gazeCleanups)root._gazeCleanups=[];
        root._gazeCleanups.push(()=>window.removeEventListener("mouseup",up));
        gCvs.addEventListener("dblclick",()=>{_setGazeForFrame(state.frame,eyeKey,0,0);if(gazeLinked)_setGazeForFrame(state.frame,eyeKey==="l"?"r":"l",0,0);_drawGimbals();render();});
        wrap.append(eyeLbl,gCvs,info);
        return {el:wrap,draw:drawG};
    }
    const gimbalL = _buildGimbal("Left","l",C.gaze_l);
    const gimbalR = _buildGimbal("Right","r",C.gaze_r);
    const linkBtn = _el("button",
        `font:9px ui-sans-serif;padding:4px 6px;border:1px solid ${C.border};border-radius:4px;cursor:pointer;` +
        `background:${C.ok_bg};color:${C.text};align-self:center;`);
    linkBtn.textContent="Link"; linkBtn.title="Link both eyes";
    linkBtn.addEventListener("click",()=>{
        gazeLinked=!gazeLinked;
        linkBtn.textContent=gazeLinked?"Link":"Split";
        linkBtn.style.background=gazeLinked?C.ok_bg:C.btn_off_bg;
    });
    ctxGaze.append(gimbalL.el,linkBtn,gimbalR.el);
    ctxPanels.gaze = ctxGaze;

    // -- Pose context: 3D editor + collapsed advanced
    const ctxPose = _el("div", _ctxPanelStyle + "display:flex;flex-direction:column;gap:2px;");
    const btn3D = _btn("3D Head Editor\u2026","Open 3D head pose (loads Three.js on demand)");
    btn3D.style.width="100%";
    let _fc3dEditor=null,_fc3dHost=null;
    ctxPose.appendChild(btn3D);
    const poseAdv = _el("details",`background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;`);
    const poseAdvSum = _el("summary",`padding:4px 8px;font:10px ui-sans-serif;color:${C.dim};cursor:pointer;`);
    poseAdvSum.textContent="\u25B8 Advanced";
    poseAdv.appendChild(poseAdvSum);
    const poseAdvRow = _el("div","display:flex;gap:4px;padding:4px 6px;flex-wrap:wrap;");
    const btnPropToggle = _btn("\u0394 off","Delta propagation for drags");
    let propagateMode = false;
    btnPropToggle.addEventListener("click",()=>{propagateMode=!propagateMode;btnPropToggle.textContent=propagateMode?"\u0394 ON":"\u0394 off";btnPropToggle.style.background=propagateMode?C.accent:C.border;});
    const btnClearAll = _btn("Clear all frames", "Clear every override", C.err_bg);
    btnClearAll.addEventListener("click",()=>{writeOverrides(node,{frames:{}});writePoseOverrides(node,{frames:{}});writeGazeOverrides(node,{frames:{}});const d=_parseCoeffsJson();d.frames={};_writeCoeffsJson(d);tlstate.selA=tlstate.selB=-1;render();drawTimeline();});
    poseAdvRow.append(btnPropToggle,btnClearAll);
    poseAdv.appendChild(poseAdvRow);
    poseAdv.addEventListener("toggle",()=>{poseAdvSum.textContent=(poseAdv.open?"\u25BE ":"\u25B8 ")+"Advanced";_relayout();});
    ctxPose.appendChild(poseAdv);
    ctxPanels.pose = ctxPose;

    // -- Settings: pipeline parameters (Comfy widgets synced in-tab; only JSON hidden)
    const ctxSet = _el("div", _ctxPanelStyle + "display:flex;flex-direction:column;gap:3px;");
    const _setSections = [];
    const _setUiSync = [];

    function _mkSetSection(title, open = true) {
        const det = _el("details",
            `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;`);
        det.open = open !== false;
        const sum = _el("summary",
            `padding:4px 8px;font:10px ui-sans-serif;color:${C.dim};cursor:pointer;user-select:none;`);
        sum.textContent = (open ? "\u25BE " : "\u25B8 ") + title;
        det.appendChild(sum);
        const body = _el("div", "display:flex;flex-direction:column;gap:2px;padding:4px 8px 6px;");
        det.appendChild(body);
        det.addEventListener("toggle", () => {
            sum.textContent = (det.open ? "\u25BE " : "\u25B8 ") + title;
            _relayout();
        });
        _setSections.push(det);
        ctxSet.appendChild(det);
        return body;
    }

    function _addFloatRow(parent, name, label) {
        const opts = FC3D_PARAM_OPTS[name] || {};
        const min = opts.min ?? -3, max = opts.max ?? 3, step = opts.step ?? 0.05;
        const row = _el("div",
            "display:grid;grid-template-columns:clamp(72px,34%,110px) 1fr minmax(32px,2.5em);" +
            "align-items:center;gap:4px;min-height:22px;width:100%;");
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};overflow:hidden;text-overflow:ellipsis;`);
        lbl.textContent = label || name;
        const sl = _el("input", `width:100%;height:10px;accent-color:${C.sel};cursor:pointer;`);
        sl.type = "range"; sl.min = String(min); sl.max = String(max); sl.step = String(step);
        sl.value = String(Number(readParam(node, name)) || 0);
        const val = _el("span", `font:9px ui-monospace,monospace;color:${C.dim};text-align:right;`);
        const _fmt = (v) => (Math.abs(step) < 0.1 ? Number(v).toFixed(2) : Number(v).toFixed(1));
        val.textContent = _fmt(sl.value);
        sl.addEventListener("input", () => {
            const v = parseFloat(sl.value);
            val.textContent = _fmt(v);
            _fc3dEmitParam(node, name, v, _onWidgetChg);
        });
        _setUiSync.push(() => {
            sl.value = String(Number(readParam(node, name)) ?? 0);
            val.textContent = _fmt(sl.value);
        });
        row.append(lbl, sl, val);
        parent.appendChild(row);
    }

    function _addComboRow(parent, name, label, values) {
        const row = _el("div",
            "display:grid;grid-template-columns:clamp(72px,34%,110px) 1fr;align-items:center;gap:4px;min-height:22px;width:100%;");
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};overflow:hidden;text-overflow:ellipsis;`);
        lbl.textContent = label || name;
        const sel = _el("select",
            `font:9px ui-sans-serif;padding:2px;background:${C.input_bg};color:${C.text};` +
            `border:1px solid ${C.border};border-radius:3px;width:100%;max-width:100%;`);
        const vals = values || PROPAGATE_OPTS;
        for (const v of vals) {
            const o = _el("option"); o.value = v; o.textContent = v;
            sel.appendChild(o);
        }
        sel.value = String(readParam(node, name) ?? vals[0]);
        sel.addEventListener("change", () => _fc3dEmitParam(node, name, sel.value, _onWidgetChg));
        _setUiSync.push(() => { sel.value = String(readParam(node, name) ?? vals[0]); });
        row.append(lbl, sel);
        parent.appendChild(row);
    }

    function _addBoolRow(parent, name, label) {
        const row = _el("div", "display:flex;align-items:center;gap:5px;height:18px;");
        const cb = _el("input", `accent-color:${C.sel};cursor:pointer;`);
        cb.type = "checkbox"; cb.checked = !!readParam(node, name);
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};cursor:pointer;`);
        lbl.textContent = label || name;
        const flip = () => _fc3dEmitParam(node, name, cb.checked, _onWidgetChg);
        cb.addEventListener("change", flip);
        lbl.addEventListener("click", () => { cb.checked = !cb.checked; flip(); });
        _setUiSync.push(() => { cb.checked = !!readParam(node, name); });
        row.append(cb, lbl);
        parent.appendChild(row);
    }

    function _addIntRow(parent, name, label) {
        const opts = FC3D_PARAM_OPTS[name] || {};
        const row = _el("div",
            "display:grid;grid-template-columns:clamp(72px,34%,110px) 1fr;align-items:center;gap:4px;min-height:22px;width:100%;");
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};overflow:hidden;text-overflow:ellipsis;`);
        lbl.textContent = label || name;
        const inp = _el("input",
            `padding:2px 4px;background:${C.input_bg};color:${C.text};border:1px solid ${C.border};` +
            `border-radius:3px;font:9px ui-monospace,monospace;width:100%;max-width:100%;box-sizing:border-box;`);
        inp.type = "number";
        if (opts.min !== undefined) inp.min = String(opts.min);
        if (opts.max !== undefined) inp.max = String(opts.max);
        if (opts.step !== undefined) inp.step = String(opts.step);
        inp.value = String(Number(readParam(node, name)) ?? 0);
        inp.addEventListener("change", () => {
            let v = parseInt(inp.value, 10);
            if (!Number.isFinite(v)) v = 0;
            _fc3dEmitParam(node, name, v, _onWidgetChg);
        });
        _setUiSync.push(() => { inp.value = String(Number(readParam(node, name)) ?? 0); });
        row.append(lbl, inp);
        parent.appendChild(row);
    }

    const bExpr = _mkSetSection("Expression & range", true);
    _addFloatRow(bExpr, "expression_strength", "Expr strength");
    _addFloatRow(bExpr, "expression_clamp", "Expr clamp");
    _addComboRow(bExpr, "propagate_expression", "Expr propagate");
    _addComboRow(bExpr, "use_metas", "Use metas", ["edited", "original"]);
    _addIntRow(bExpr, "frame_start", "Frame start");
    _addIntRow(bExpr, "frame_end", "Frame end");

    const bHead = _mkSetSection("Head pose", true);
    _addFloatRow(bHead, "head_yaw_deg", "Yaw");
    _addFloatRow(bHead, "head_pitch_deg", "Pitch");
    _addFloatRow(bHead, "head_roll_deg", "Roll");
    _addFloatRow(bHead, "head_tx", "Shift X");
    _addFloatRow(bHead, "head_ty", "Shift Y");
    _addFloatRow(bHead, "head_tz", "Depth");
    _addFloatRow(bHead, "head_scale", "Scale");
    _addFloatRow(bHead, "jaw_rot_deg", "Jaw");
    _addFloatRow(bHead, "neck_yaw_deg", "Neck yaw");
    _addFloatRow(bHead, "neck_pitch_deg", "Neck pitch");
    _addComboRow(bHead, "propagate_head", "Head propagate");

    const bGaze = _mkSetSection("Gaze (global)", true);
    _addFloatRow(bGaze, "gaze_yaw_deg", "Gaze yaw");
    _addFloatRow(bGaze, "gaze_pitch_deg", "Gaze pitch");
    _addComboRow(bGaze, "propagate_gaze", "Gaze propagate");

    const bBlend = _mkSetSection("Reference blend", true);
    _addFloatRow(bBlend, "blend_strength", "Blend strength");
    _addBoolRow(bBlend, "blend_mouth", "Mouth");
    _addBoolRow(bBlend, "blend_brows", "Brows");
    _addBoolRow(bBlend, "blend_eyes", "Eyes");
    _addBoolRow(bBlend, "blend_jaw", "Jaw");
    const blendHint = _el("span", `font:9px ui-sans-serif;color:${C.dim};padding:0 2px;`);
    blendHint.textContent = "Connect reference_pose_data on the node socket.";
    bBlend.appendChild(blendHint);

    const bPrev = _mkSetSection("Preview output", true);
    _addIntRow(bPrev, "preview_frame_idx", "Preview frame");
    _addIntRow(bPrev, "preview_size", "Preview px");
    _addIntRow(bPrev, "preview_max_video_frames", "Max vid frames");

    const setHint = _el("div",
        `font:9px ui-sans-serif;color:${C.dim};padding:2px 4px 4px;line-height:1.35;`);
    setHint.textContent =
        "Sliders and options here mirror the node widgets (workflow saves them). " +
        "Open sections below or scroll. Canvas edits write JSON overrides only.";
    ctxSet.insertBefore(setHint, ctxSet.firstChild);

    ctxPanels.set = ctxSet;

    function _syncSettingsFromWidgets() {
        for (const fn of _setUiSync) {
            try { fn(); } catch (_) {}
        }
    }

    // Scrollable context strip — parameters stay reachable without clipping
    const ctxScroll = _el("div",
        `flex:0 1 auto;min-height:0;max-height:min(${FC3D_CTX_SCROLL_MAX}px,42vh);` +
        `overflow-y:auto;overflow-x:hidden;pointer-events:auto;` +
        `-webkit-overflow-scrolling:touch;`);
    for (const [id, panel] of Object.entries(ctxPanels)) {
        panel.style.display = id === "face" ? "flex" : "none";
        panel.style.flexDirection = "column";
        ctxScroll.appendChild(panel);
    }
    root.appendChild(ctxScroll);

    let _resizeObs = null;
    try {
        _resizeObs = new ResizeObserver(() => {
            if (_syncCanvasPx()) {
                try { render(); drawTimeline(); } catch (_) {}
            }
        });
        _resizeObs.observe(canvasWrap);
    } catch (_) {}

    // ── Dynamic node height (measured, tab-aware, debounced setSize) ─
    let _editorH = 400;
    let _lastLayoutH = 0;
    let _relayoutTimer = 0;
    const _CHROME_PX = 52 + 54 + 22 + 12;

    /** Grow canvas + context strip to consume user-resized node height (no dead band). */
    function _fillAvailableHeight(wantW, wantH) {
        const wantHClamped = Math.max(FC3D_MIN_H, Math.min(FC3D_MAX_H, wantH || _editorH));
        const innerW = Math.max(CANVAS_MIN_PX, Math.min(CANVAS_MAX_PX, (wantW || FC3D_NODE_W) - 52));
        _applyCanvasViewPx(innerW);

        const panel = ctxPanels[activeTab];
        let ctxMin = 48;
        if (panel && panel.style.display !== "none") {
            const tabCap = FC3D_TAB_CTX_MAX[activeTab] ?? FC3D_CTX_SCROLL_MAX;
            ctxMin = Math.min(tabCap, Math.max(48, panel.scrollHeight + 4));
        }
        const canvasBudget = wantHClamped - _CHROME_PX - ctxMin;
        if (canvasBudget > _canvasViewPx + 8) {
            _applyCanvasViewPx(Math.min(CANVAS_MAX_PX, canvasBudget));
        }
        const ctxBudget = Math.max(ctxMin, wantHClamped - _CHROME_PX - _canvasViewPx);
        if (ctxScroll) {
            ctxScroll.style.maxHeight = `${Math.min(FC3D_MAX_H, ctxBudget)}px`;
            ctxScroll.style.flex = ctxBudget > ctxMin + 8 ? "1 1 auto" : "0 1 auto";
        }

        _editorH = wantHClamped;
        root.style.height = `${wantHClamped}px`;
        root.style.minHeight = `${Math.min(wantHClamped, _canvasViewPx + _CHROME_PX + ctxMin)}px`;
        root.style.maxHeight = `${FC3D_MAX_H}px`;
        return wantHClamped;
    }

    function _relayout() {
        let ctxH = 0;
        const panel = ctxPanels[activeTab];
        if (panel && panel.style.display !== "none") {
            const tabCap = FC3D_TAB_CTX_MAX[activeTab] ?? FC3D_CTX_SCROLL_MAX;
            const minH = activeTab === "set" ? 120 : 48;
            ctxH = Math.min(tabCap, Math.max(minH, panel.scrollHeight + 4));
        }
        const computedMin = Math.min(FC3D_MAX_H, Math.max(FC3D_MIN_H, _canvasViewPx + _CHROME_PX - 12 + ctxH + 12));

        _applyCanvasViewPx(FC3D_TAB_CANVAS_PX[activeTab] ?? FC3D_CANVAS_VIEW_PX);
        _editorH = computedMin;
        root.style.height = `${_editorH}px`;
        root.style.minHeight = `${FC3D_MIN_H}px`;
        root.style.maxHeight = `${FC3D_MAX_H}px`;
        if (ctxScroll) {
            ctxScroll.style.maxHeight = `min(${FC3D_CTX_SCROLL_MAX}px, 42vh)`;
            ctxScroll.style.flex = "0 1 auto";
        }
        if (Math.abs(_editorH - _lastLayoutH) > 8) {
            _lastLayoutH = _editorH;
            try { _fc3dSyncNodeSize(node); } catch (_) {}
        }
    }
    function _scheduleRelayout() {
        if (_relayoutTimer) clearTimeout(_relayoutTimer);
        _relayoutTimer = setTimeout(() => { _relayoutTimer = 0; _relayout(); }, 40);
    }

    // ── Tab switching ───────────────────────────────────────────────
    function switchTab(id) {
        activeTab = id;
        for (const t of TABS) {
            tabBtns[t.id].style.background = t.id===id ? C.tab_active : "transparent";
            if (ctxPanels[t.id]) ctxPanels[t.id].style.display = t.id === id ? "flex" : "none";
        }
        const tabCanvas = FC3D_TAB_CANVAS_PX[id] ?? FC3D_CANVAS_VIEW_PX;
        if (Math.abs(tabCanvas - _canvasViewPx) > 4) {
            _applyCanvasViewPx(tabCanvas);
        }
        if (id === "expr") _refreshSliders();
        if (id === "gaze") _drawGimbals();
        if (id === "set") _syncSettingsFromWidgets();
        _relayout();
        render();
        if (id === "gaze") _drawGimbals();
        try{_persistSave?.();}catch(_){}
    }

    // ── Undo/Redo system ────────────────────────────────────────────
    const _undoStack=[],_redoStack=[],_UNDO_LIMIT=50;
    function _snapshotState(){const snap={};for(const wn of[OVERRIDE_WIDGET,POSE_OVERRIDE_WIDGET,GAZE_OVERRIDE_WIDGET]){const w=readWidget(node,wn);if(w)snap[wn]=w.value||"";}snap[COEFF_WIDGET]=String(readParam(node,COEFF_WIDGET)||"");return JSON.stringify(snap);}
    function _pushUndo(){const s=_snapshotState();if(_undoStack.length>0&&_undoStack[_undoStack.length-1]===s)return;_undoStack.push(s);if(_undoStack.length>_UNDO_LIMIT)_undoStack.shift();_redoStack.length=0;_updateUndoButtons();}
    function _restoreSnap(snap){try{const obj=JSON.parse(snap);for(const[wn,val]of Object.entries(obj)){if(wn===COEFF_WIDGET){writeParam(node,wn,val);continue;}const w=readWidget(node,wn);if(w){w.value=val;try{w.callback?.(val,node,w);}catch(_){}}}}catch(_){}}
    function _undo(){if(_undoStack.length<2)return;_redoStack.push(_undoStack.pop());_restoreSnap(_undoStack[_undoStack.length-1]);_updateUndoButtons();render();}
    function _redo(){if(!_redoStack.length)return;const s=_redoStack.pop();_undoStack.push(s);_restoreSnap(s);_updateUndoButtons();render();}
    function _updateUndoButtons(){btnUndo.disabled=_undoStack.length<2;btnRedo.disabled=!_redoStack.length;}
    _undoStack.push(_snapshotState());
    let _undoTimer=null;
    const _scheduleUndoSnap=()=>{if(_undoTimer)clearTimeout(_undoTimer);_undoTimer=setTimeout(()=>{_pushUndo();_undoTimer=null;},300);};
    btnUndo.addEventListener("click",_undo);
    btnRedo.addEventListener("click",_redo);

    // ── Core state ──────────────────────────────────────────────────
    const state = { meta:makeDefaultMeta(), frame:0, dragLm:-1, hoverLm:-1, selTarget:{kind:"face",idx:30} };
    const gstate = { dragEye:null, hoverEye:null };
    const pstate = { edges:POSE18_EDGES_DEFAULT, names:POSE18_NAMES, frames:[], dragJ:-1, hoverJ:-1 };
    const tlstate = { selA:-1, selB:-1, hover:-1 };

    function selectedSet(){return new Set(state.meta?.selected||[]);}
    function emphSet(){return new Set(state.meta?.eye_emph||[]);}
    function _frameCount(){return Math.max(0,Number(slider.max)+1);}

    // ── Expression coefficient helpers (live draft + debounced widget commit) ──
    let _coeffDraft = null;
    let _coeffCommitT = 0;
    function _parseCoeffsJson() {
        if (_coeffDraft) return _coeffDraft;
        const raw = readParam(node, COEFF_WIDGET);
        if (!raw) return { frames: {}, ranges: [] };
        try {
            const o = JSON.parse(String(raw));
            if (!o.frames || typeof o.frames !== "object") o.frames = {};
            if (!Array.isArray(o.ranges)) o.ranges = [];
            return o;
        } catch (_) {
            return { frames: {}, ranges: [] };
        }
    }
    function _flushCoeffDraft() {
        if (!_coeffDraft) return;
        _scheduleUndoSnap();
        writeParam(node, COEFF_WIDGET, JSON.stringify(_coeffDraft, null, 0));
        _coeffDraft = null;
    }
    function _scheduleCoeffCommit() {
        if (_coeffCommitT) clearTimeout(_coeffCommitT);
        _coeffCommitT = setTimeout(() => { _coeffCommitT = 0; _flushCoeffDraft(); }, 80);
    }
    function _writeCoeffsJson(obj) {
        _coeffDraft = obj;
        _flushCoeffDraft();
        render();
        _scheduleServerResync();
    }
    function _coeffsForFrame(fi) {
        const data = _parseCoeffsJson(), out = {};
        FACS_AXES.forEach((a) => { out[a.id] = 0; });
        if (data.frames[String(fi)]) Object.assign(out, data.frames[String(fi)]);
        return out;
    }
    function _setCoeff(fi, axisId, val) {
        const data = _parseCoeffsJson();
        if (!data.frames[String(fi)]) data.frames[String(fi)] = {};
        if (Math.abs(val) < 0.001) {
            delete data.frames[String(fi)][axisId];
            if (!Object.keys(data.frames[String(fi)]).length) delete data.frames[String(fi)];
        } else {
            data.frames[String(fi)][axisId] = +val.toFixed(3);
        }
        _coeffDraft = data;
        render();
        _scheduleCoeffCommit();
        _scheduleServerResync();
    }
    function _applyPreset(name) {
        const preset = EMOTION_PRESETS[name];
        if (!preset) return;
        const fi = state.frame, data = _parseCoeffsJson();
        data.frames[String(fi)] = {};
        for (const [k, v] of Object.entries(preset)) {
            if (Math.abs(v) >= 0.001) data.frames[String(fi)][k] = +v.toFixed(3);
        }
        _writeCoeffsJson(data);
        _refreshSliders();
    }
    function _refreshSliders(){const coeffs=_coeffsForFrame(state.frame);for(const a of FACS_AXES){const el=_sliderEls[a.id];if(!el)continue;const v=coeffs[a.id]||0;el.sl.value=String(v);el.val.textContent=v.toFixed(2);el.val.style.color=Math.abs(v)>0.01?C.sel:C.dim;}}
    // ── Gaze helpers ────────────────────────────────────────────────
    function _getGazeForFrame(fi,eye){const data=parseGazeOverrides(node),fr=data.frames?.[String(fi)];if(fr&&fr[eye])return{yaw:(fr[eye][0]||0)*(180/Math.PI),pitch:(fr[eye][1]||0)*(180/Math.PI)};return{yaw:0,pitch:0};}
    function _setGazeForFrame(fi,eye,yawDeg,pitchDeg){const ov=parseGazeOverrides(node),key=String(fi);if(!ov.frames[key])ov.frames[key]={};const yr=yawDeg*(Math.PI/180),pr=pitchDeg*(Math.PI/180);if(Math.abs(yr)<0.001&&Math.abs(pr)<0.001){delete ov.frames[key][eye];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];}else ov.frames[key][eye]=[+yr.toFixed(5),+pr.toFixed(5)];writeGazeOverrides(node,ov);}
    function _drawGimbals(){gimbalL.draw();gimbalR.draw();}

    // ── Gaze handle math (on main canvas) ───────────────────────────
    const _eyeRange=eye=>eye==="r"?[36,42]:[42,48];
    function _eyeCentroid(lms,eye){const[a,b]=_eyeRange(eye);let sx=0,sy=0,n=0;for(let i=a;i<b;i++){sx+=lms[i][0];sy+=lms[i][1];n++;}return[sx/n,sy/n];}
    const _GAZE_OX=0.10,_GAZE_OY=0.06;
    function _currentGazeForEye(f,eye){const ov=parseGazeOverrides(node),fov=ov.frames?.[String(f)]?.[eye];if(Array.isArray(fov)&&fov.length===2)return{yaw:Number(fov[0])||0,pitch:Number(fov[1])||0};const gf=Array.isArray(state.meta?.gaze?.frames)?state.meta.gaze.frames[f]:null;const pair=gf&&(eye==="l"?gf.l_gaze:gf.r_gaze);if(Array.isArray(pair)&&pair.length===2)return{yaw:Number(pair[0])||0,pitch:Number(pair[1])||0};return{yaw:0,pitch:0};}
    function _gazeMax(){return{yaw:Number(state.meta?.gaze?.max_yaw_rad)||(30*Math.PI/180),pitch:Number(state.meta?.gaze?.max_pitch_rad)||(25*Math.PI/180)};}
    function computeGazeHandles(lms,W,H){if(!state.meta?.gaze)return[];const max=_gazeMax(),out=[];for(const eye of["l","r"]){const[cxN,cyN]=_eyeCentroid(lms,eye);const[ax,ay]=denormToCanvas(cxN,cyN,W,H);const{yaw,pitch}=_currentGazeForEye(state.frame,eye);const yN=Math.max(-1,Math.min(1,yaw/max.yaw)),pN=Math.max(-1,Math.min(1,pitch/max.pitch));out.push({eye,ax,ay,tx:ax+yN*_GAZE_OX*W,ty:ay-pN*_GAZE_OY*H});}return out;}
    function pickGazeHandle(mx,my){if(!state.meta?.gaze)return null;const lms=landmarksForFrame(state.frame),handles=computeGazeHandles(lms,cvs.width,cvs.height);let best=null,bestD=100;for(const h of handles){const d2=(mx-h.tx)**2+(my-h.ty)**2;if(d2<bestD){bestD=d2;best=h.eye;}}return best;}
    function canvasToGaze(mx,my,eye){const lms=landmarksForFrame(state.frame),[cxN,cyN]=_eyeCentroid(lms,eye),[ax,ay]=denormToCanvas(cxN,cyN,cvs.width,cvs.height),max=_gazeMax();const yN=(mx-ax)/Math.max(1,_GAZE_OX*cvs.width),pN=-(my-ay)/Math.max(1,_GAZE_OY*cvs.height);return{yaw:Math.max(-max.yaw,Math.min(max.yaw,yN*max.yaw)),pitch:Math.max(-max.pitch,Math.min(max.pitch,pN*max.pitch))};}

    // ── Landmark computation ────────────────────────────────────────
    function _accumRangeDelta(ov,idx,f){let dx=0,dy=0;const N=_frameCount(),ranges=Array.isArray(ov.ranges)?ov.ranges:[];for(const en of ranges){if(!en||!en.delta)continue;const s=Math.max(0,Number(en.start)||0);let e=Number(en.end);if(!(e>=0))e=Math.max(0,N-1);if(f<s||f>e)continue;const d=en.delta[String(idx)];if(Array.isArray(d)&&d.length===2){dx+=Number(d[0])||0;dy+=Number(d[1])||0;}}return[dx,dy];}
    function _ensureRangeEntry(ov,s,e){if(!Array.isArray(ov.ranges))ov.ranges=[];let entry=ov.ranges.find(en=>en&&Number(en.start)===s&&Number(en.end)===e);if(!entry){entry={start:s,end:e,delta:{}};ov.ranges.push(entry);}if(!entry.delta||typeof entry.delta!=="object")entry.delta={};return entry;}
    function _propRange(){const r=_tlRange();return r?[r[0],r[1]]:[0,-1];}
    function _tlRange(){if(tlstate.selA<0||tlstate.selB<0)return null;return[Math.min(tlstate.selA,tlstate.selB),Math.max(tlstate.selA,tlstate.selB)];}

    function _faceBBoxSize(lms) {
        let mnX = 1, mxX = 0, mnY = 1, mxY = 0;
        for (const p of lms) {
            mnX = Math.min(mnX, p[0]); mxX = Math.max(mxX, p[0]);
            mnY = Math.min(mnY, p[1]); mxY = Math.max(mxY, p[1]);
        }
        return [Math.max(1e-6, mxX - mnX), Math.max(1e-6, mxY - mnY)];
    }

    function _applyExpressionCoeffs(lms, coeffs) {
        const strn = Number(readParam(node, "expression_strength")) || 1;
        const clamp = Number(readParam(node, "expression_clamp")) || 1.5;
        const [bw, bh] = _faceBBoxSize(lms);
        const out = lms.map((p) => [p[0], p[1]]);
        for (const axis of FACS_AXES) {
            const raw = coeffs[axis.id] || 0;
            const v = Math.max(-clamp, Math.min(clamp, raw * strn));
            if (Math.abs(v) < 1e-6) continue;
            const tab = FACS_BASIS[axis.id];
            if (!tab) continue;
            for (const [idx, d] of Object.entries(tab)) {
                const i = parseInt(idx, 10);
                if (i >= 0 && i < out.length && Array.isArray(d)) {
                    out[i][0] += d[0] * bw * v;
                    out[i][1] += d[1] * bh * v;
                }
            }
        }
        return out;
    }

    function _baseLandmarksForFrame(f) {
        const fr = state.meta?.frames?.[f];
        if (fr?.lms && Array.isArray(fr.lms) && fr.lms.length >= 68) {
            return fr.lms.slice(0, 68).map((p) => [
                Number(Array.isArray(p) ? p[0] : 0) || 0,
                Number(Array.isArray(p) ? p[1] : 0) || 0,
            ]);
        }
        const bl = _localBaseline?.faceLm?.[f];
        if (bl && bl.length === 68) return bl.map((p) => [p[0], p[1]]);
        return CANONICAL.map((p) => [p[0], p[1]]);
    }

    function landmarksForFrame(f) {
        let lms = _baseLandmarksForFrame(f);
        lms = _applyExpressionCoeffs(lms, _coeffsForFrame(f));
        if (state.meta?.d_norm && Array.isArray(state.meta.d_norm)) {
            const sel = selectedSet(), stren = state.meta.strength ?? 0.7;
            for (let i = 0; i < lms.length && i < state.meta.d_norm.length; i++) {
                if (sel.has(i)) {
                    lms[i][0] += state.meta.d_norm[i][0] * stren;
                    lms[i][1] += state.meta.d_norm[i][1] * stren;
                }
            }
        }
        const ov = parseOverrides(node);
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
                    lms[i][0] = v[0]; lms[i][1] = v[1];
                }
            }
        }
        return lms;
    }

    // ── Video underlay ──────────────────────────────────────────────
    let _videoFrames=[];
    function _setVideoFrames(urls){_videoFrames=[];for(const url of(urls||[])){const img=new Image();img.src=url;_videoFrames.push(img);}}

    // ── Rendering functions ─────────────────────────────────────────
    let _tlRafPending = false;
    function render() {
        if(!_tlRafPending){_tlRafPending=true;requestAnimationFrame(()=>{_tlRafPending=false;drawTimeline();});}
        const ctx=cvs.getContext("2d"),W=cvs.width,H=cvs.height;
        ctx.fillStyle=C.canvas_bg;ctx.fillRect(0,0,W,H);
        switch(activeTab) {
            case "face": case "gaze": case "set":
                _renderFaceGaze(ctx,W,H,activeTab==="gaze"); break;
            case "expr": _renderExpr(ctx,W,H); break;
            case "pose": _renderPose(ctx,W,H); break;
        }
    }

    function _drawGrid(ctx,W,H){ctx.strokeStyle=C.grid;ctx.lineWidth=1;for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo((i/4)*W,0);ctx.lineTo((i/4)*W,H);ctx.stroke();ctx.beginPath();ctx.moveTo(0,(i/4)*H);ctx.lineTo(W,(i/4)*H);ctx.stroke();}}
    function _drawVideoUnderlay(ctx,W,H){if(_videoFrames.length>0){const fi=Math.min(state.frame,_videoFrames.length-1),img=_videoFrames[fi];if(img&&img.complete&&img.naturalWidth>0){ctx.globalAlpha=0.3;ctx.drawImage(img,0,0,W,H);ctx.globalAlpha=1;}}}
    function _drawFaceWireframe(ctx,W,H,lms,alpha){
        ctx.globalAlpha=alpha||1;ctx.strokeStyle=C.other;ctx.lineWidth=1.5;
        for(const seg of SEGMENTS){const[a,b,closed]=seg;ctx.beginPath();for(let i=a;i<b;i++){const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);if(i===a)ctx.moveTo(x,y);else ctx.lineTo(x,y);}if(closed)ctx.closePath();ctx.stroke();}
        ctx.globalAlpha=1;
    }

    function _renderFaceGaze(ctx,W,H,gazeEmphasis) {
        _drawVideoUnderlay(ctx,W,H);_drawGrid(ctx,W,H);
        if (state.meta?._synthetic) {
            ctx.fillStyle = C.dim; ctx.font="10px ui-sans-serif"; ctx.textAlign = "left";
            ctx.fillText("Queue once with pose_data for real landmarks", 4, H - 4);
        }
        const lms=landmarksForFrame(state.frame),sel=selectedSet(),emp=emphSet();
        _drawFaceWireframe(ctx,W,H,lms,gazeEmphasis?0.4:1);
        for(let i=0;i<lms.length;i++){const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);const isSel=sel.has(i),isEmph=emp.has(i);const r=(state.hoverLm===i||state.dragLm===i)?6:isSel?4:2.5;ctx.beginPath();ctx.fillStyle=gazeEmphasis?(C.other+"80"):(isEmph?C.emph:(isSel?C.sel:C.other));ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
        if(!gazeEmphasis&&state.hoverLm>=0){const[x,y]=denormToCanvas(lms[state.hoverLm][0],lms[state.hoverLm][1],W,H);ctx.fillStyle=C.text;ctx.font="10px ui-monospace,monospace";ctx.textAlign="left";ctx.fillText(String(state.hoverLm),x+6,y-4);}
        // Gaze handles
        const handles=computeGazeHandles(lms,W,H);
        for(const h of handles){const isDrag=gstate.dragEye===h.eye,isHov=gstate.hoverEye===h.eye,col=isDrag?C.gaze_drag:(h.eye==="l"?C.gaze_l:C.gaze_r);
            ctx.strokeStyle=col;ctx.lineWidth=gazeEmphasis?2.5:2;ctx.beginPath();ctx.arc(h.ax,h.ay,gazeEmphasis?5:4,0,Math.PI*2);ctx.stroke();
            ctx.beginPath();ctx.moveTo(h.ax,h.ay);ctx.lineTo(h.tx,h.ty);ctx.stroke();
            const r=isDrag?7:(isHov?6:(gazeEmphasis?5:4));ctx.fillStyle=col;ctx.beginPath();ctx.arc(h.tx,h.ty,r,0,Math.PI*2);ctx.fill();}
    }

    function _renderExpr(ctx,W,H) {
        _drawGrid(ctx,W,H);
        const lms=landmarksForFrame(state.frame);
        _drawFaceWireframe(ctx,W,H,lms,1);
        // Blendshape heat overlay: colorize face regions based on active coefficients
        const coeffs=_coeffsForFrame(state.frame);
        const regions=[
            {ids:[17,18,19,20,21,22,23,24,25,26],keys:["brow_inner_raise","brow_outer_raise","brow_furrow"]},
            {ids:[36,37,38,39,40,41,42,43,44,45,46,47],keys:["eye_close_L","eye_close_R","cheek_raise"]},
            {ids:[27,28,29,30,31,32,33,34,35],keys:["nose_wrinkle"]},
            {ids:[48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67],keys:["smile","frown","mouth_open","jaw_drop","lip_pucker"]},
        ];
        for(const reg of regions){
            let maxAbs=0;
            for(const k of reg.keys) maxAbs=Math.max(maxAbs,Math.abs(coeffs[k]||0));
            if(maxAbs<0.02) continue;
            const intensity=Math.min(1,maxAbs);
            ctx.fillStyle=maxAbs>0?C.sel:C.gaze_l;
            ctx.globalAlpha=intensity*0.5;
            for(const i of reg.ids){if(i>=lms.length)continue;const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();}
            ctx.globalAlpha=1;
        }
        // Active coefficient summary
        const active=Object.entries(coeffs).filter(([_,v])=>Math.abs(v)>0.01);
        if(active.length){
            ctx.font="9px ui-sans-serif";ctx.textAlign="left";
            const top=active.sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,3);
            let ty=12;
            for(const[k,v]of top){ctx.fillStyle=v>0?C.sel:C.gaze_l;ctx.fillText(`${k} ${v>0?"+":""}${v.toFixed(2)}`,4,ty);ty+=10;}
        }
    }

    function _renderPose(ctx,W,H) {
        _drawGrid(ctx,W,H);
        if(!pstate.frames.length){ctx.fillStyle=C.dim;ctx.font="11px ui-sans-serif";ctx.textAlign="center";ctx.fillText("no pose data \u2014 queue once",W/2,H/2);return;}
        const f=state.frame,kps=_bodyKpsForFrame(f);
        if(!kps){ctx.fillStyle=C.dim;ctx.font="11px ui-sans-serif";ctx.textAlign="center";ctx.fillText("no body keypoints for frame "+f,W/2,H/2);return;}
        const fr=pstate.frames[f],imgW=Math.max(1,Number(fr.w)||1),imgH=Math.max(1,Number(fr.h)||1);
        const scale=Math.min(W/imgW,H/imgH),drawW=imgW*scale,drawH=imgH*scale,ox=(W-drawW)/2,oy=(H-drawH)/2;
        ctx.strokeStyle=C.border;ctx.lineWidth=1;ctx.strokeRect(ox+0.5,oy+0.5,drawW-1,drawH-1);
        const _toCvs=(xn,yn)=>[ox+xn*drawW,oy+yn*drawH];
        ctx.strokeStyle=C.pose_bone;ctx.lineWidth=2.5;
        for(const[a,b]of pstate.edges){const ka=kps[a],kb=kps[b];if(!ka||!kb)continue;const[x1,y1]=_toCvs(ka[0],ka[1]),[x2,y2]=_toCvs(kb[0],kb[1]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}
        for(let i=0;i<kps.length;i++){const k=kps[i];if(!k)continue;const[x,y]=_toCvs(k[0],k[1]);const isDrag=pstate.dragJ===i,isHover=pstate.hoverJ===i,r=isDrag?7:(isHover?6:4.5);ctx.beginPath();ctx.fillStyle=isDrag?C.pose_selected:C.pose_joint;ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.lineWidth=1;ctx.strokeStyle=C.canvas_bg;ctx.stroke();}
        if(pstate.hoverJ>=0&&kps[pstate.hoverJ]){const[x,y]=_toCvs(kps[pstate.hoverJ][0],kps[pstate.hoverJ][1]);ctx.fillStyle=C.text;ctx.font="10px ui-monospace,monospace";ctx.textAlign="left";ctx.fillText(`${pstate.hoverJ}:${pstate.names[pstate.hoverJ]||"?"}`,x+7,y-5);}
    }

    function _bodyKpsForFrame(f) {
        const fr=pstate.frames[f];if(!fr||!fr.ok||!Array.isArray(fr.kps))return null;
        const out=fr.kps.map(p=>Array.isArray(p)?[p[0],p[1]]:null);
        const ov=parsePoseOverrides(node);
        if(Array.isArray(ov.ranges)&&ov.ranges.length){for(let i=0;i<out.length;i++){if(!out[i])continue;const[dx,dy]=_accumRangeDelta(ov,i,f);if(dx||dy)out[i]=[Math.max(0,Math.min(1,out[i][0]+dx)),Math.max(0,Math.min(1,out[i][1]+dy))];}}
        const f_ov=ov.frames?.[String(f)];if(f_ov){for(const[k,v]of Object.entries(f_ov)){const i=parseInt(k,10);if(i>=0&&i<out.length&&Array.isArray(v)&&v.length===2)out[i]=[Number(v[0]),Number(v[1])];}}
        return out;
    }
    function _get_body_kps_safe(fr){if(!fr)return null;if(Array.isArray(fr.keypoints_body))return fr.keypoints_body;if(Array.isArray(fr.kps_body))return fr.kps_body;if(Array.isArray(fr.body))return fr.body;return null;}

    // ── Timeline drawing ────────────────────────────────────────────
    function _editFlagsForFrame(f){const k=String(f);return{face:!!(parseOverrides(node).frames?.[k]&&Object.keys(parseOverrides(node).frames[k]).length),pose:!!(parsePoseOverrides(node).frames?.[k]&&Object.keys(parsePoseOverrides(node).frames[k]).length),gaze:!!(parseGazeOverrides(node).frames?.[k]&&Object.keys(parseGazeOverrides(node).frames[k]).length),expr:!!(_parseCoeffsJson().frames?.[k]&&Object.values(_parseCoeffsJson().frames[k]).some(v=>Math.abs(v)>0.001))};}
    function _frameToX(f,W){const n=Math.max(1,_frameCount());return(f/Math.max(1,n-1))*(W-4)+2;}
    function _xToFrame(x,W){const n=Math.max(1,_frameCount()),t=(x-2)/Math.max(1,W-4);return Math.max(0,Math.min(n-1,Math.round(t*Math.max(1,n-1))));}

    function drawTimeline() {
        const ctx=tl.getContext("2d"),W=tl.width,H=tl.height;
        ctx.fillStyle=C.canvas_bg;ctx.fillRect(0,0,W,H);
        const n=_frameCount();
        if(n<=0){ctx.fillStyle=C.dim;ctx.font="9px ui-sans-serif";ctx.textAlign="center";ctx.fillText("queue to populate",W/2,H/2+3);return;}
        const range=_tlRange();
        if(range){const xa=_frameToX(range[0],W),xb=_frameToX(range[1],W);ctx.fillStyle="rgba(137,180,250,0.18)";ctx.fillRect(Math.min(xa,xb)-1,1,Math.abs(xb-xa)+3,H-2);}
        ctx.strokeStyle=C.border;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,H-3);ctx.lineTo(W,H-3);ctx.stroke();
        const rowY=[H*0.25,H*0.50,H*0.70],rowC=[C.sel,C.pose_joint,C.gaze_l];
        for(let f=0;f<n;f++){const x=_frameToX(f,W),fl=_editFlagsForFrame(f);
            if(fl.face){ctx.fillStyle=rowC[0];ctx.fillRect(x-0.5,rowY[0]-1,1.5,2);}
            if(fl.pose){ctx.fillStyle=rowC[1];ctx.fillRect(x-0.5,rowY[1]-1,1.5,2);}
            if(fl.gaze||fl.expr){ctx.fillStyle=fl.gaze?rowC[2]:C.accent;ctx.fillRect(x-0.5,rowY[2]-1,1.5,2);}}
        if(tlstate.hover>=0&&tlstate.hover<n){const x=_frameToX(tlstate.hover,W);ctx.strokeStyle=C.dim;ctx.setLineDash([2,2]);ctx.beginPath();ctx.moveTo(x,1);ctx.lineTo(x,H-1);ctx.stroke();ctx.setLineDash([]);}
        const xp=_frameToX(state.frame,W);ctx.strokeStyle=C.accent;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(xp,1);ctx.lineTo(xp,H-1);ctx.stroke();
    }

    // ── Timeline interaction ────────────────────────────────────────
    function _tlClient(ev){const r=tl.getBoundingClientRect();return[(ev.clientX-r.left)*(tl.width/r.width)];}
    tl.addEventListener("click",ev=>{if(_frameCount()<=0)return;const[mx]=_tlClient(ev),f=_xToFrame(mx,tl.width);if(ev.shiftKey){if(tlstate.selA<0)tlstate.selA=state.frame;tlstate.selB=f;}else{tlstate.selA=tlstate.selB=-1;}_gotoFrame(f);});
    tl.addEventListener("mousemove",ev=>{if(_frameCount()<=0)return;const[mx]=_tlClient(ev),f=_xToFrame(mx,tl.width);if(f!==tlstate.hover){tlstate.hover=f;drawTimeline();}});
    tl.addEventListener("mouseleave",()=>{if(tlstate.hover!==-1){tlstate.hover=-1;drawTimeline();}});
    tl.addEventListener("contextmenu",ev=>{if(_frameCount()<=0)return;ev.preventDefault();const[mx]=_tlClient(ev),f=_xToFrame(mx,tl.width);_clearFrame(f);render();drawTimeline();});
    function _clearFrame(f){const k=String(f);const o1=parseOverrides(node),o2=parsePoseOverrides(node),o3=parseGazeOverrides(node);if(o1.frames?.[k]){delete o1.frames[k];writeOverrides(node,o1);}if(o2.frames?.[k]){delete o2.frames[k];writePoseOverrides(node,o2);}if(o3.frames?.[k]){delete o3.frames[k];writeGazeOverrides(node,o3);}}

    // ── Canvas interaction (tab-aware, pointer + zoom-correct coords) ─
    function clientToCanvas(ev) { return eventCanvas(ev); }
    function pickLandmark(mx,my){if(!state.meta)return-1;const lms=landmarksForFrame(state.frame),sel=selectedSet(),W=cvs.width,H=cvs.height;let best=-1,bestD=64;for(let i=0;i<lms.length;i++){if(!sel.has(i))continue;const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);const d2=(mx-x)**2+(my-y)**2;if(d2<bestD){bestD=d2;best=i;}}return best;}
    function _pickJoint(mx,my){if(!pstate.frames.length)return-1;const f=state.frame,kps=_bodyKpsForFrame(f);if(!kps)return-1;const fr=pstate.frames[f],imgW=Math.max(1,Number(fr.w)||1),imgH=Math.max(1,Number(fr.h)||1),W=cvs.width,H=cvs.height,scale=Math.min(W/imgW,H/imgH),drawW=imgW*scale,drawH=imgH*scale,ox=(W-drawW)/2,oy=(H-drawH)/2;let best=-1,bestD=100;for(let i=0;i<kps.length;i++){if(!kps[i])continue;const x=ox+kps[i][0]*drawW,y=oy+kps[i][1]*drawH;const d2=(mx-x)**2+(my-y)**2;if(d2<bestD){bestD=d2;best=i;}}return best;}
    function _poseCanvasToNorm(cx,cy){const f=state.frame,fr=pstate.frames[f];if(!fr)return[0,0];const imgW=Math.max(1,Number(fr.w)||1),imgH=Math.max(1,Number(fr.h)||1),W=cvs.width,H=cvs.height,scale=Math.min(W/imgW,H/imgH),drawW=imgW*scale,drawH=imgH*scale,ox=(W-drawW)/2,oy=(H-drawH)/2;return[Math.max(0,Math.min(1,(cx-ox)/Math.max(1,drawW))),Math.max(0,Math.min(1,(cy-oy)/Math.max(1,drawH)))];}

    cvs.addEventListener("mousedown",ev=>{
        const[mx,my]=clientToCanvas(ev);
        if(activeTab==="face"||activeTab==="gaze"){
            const eye=pickGazeHandle(mx,my);
            if(eye){gstate.dragEye=eye;ev.stopPropagation();ev.preventDefault();render();return;}
            if(activeTab==="face"){
                state.dragLm=pickLandmark(mx,my);
                if(state.dragLm>=0){
                    if(propagateMode){const ref=landmarksForFrame(state.frame)[state.dragLm];state._dragRef=[ref[0],ref[1]];const[s,e]=_propRange();state._dragRange=[s,e];const ov=parseOverrides(node),entry=_ensureRangeEntry(ov,s,e),cur=entry.delta[String(state.dragLm)];state._dragBase=Array.isArray(cur)&&cur.length===2?[Number(cur[0])||0,Number(cur[1])||0]:[0,0];}
                    ev.stopPropagation();ev.preventDefault();}
            }
        } else if(activeTab==="pose"){
            pstate.dragJ=_pickJoint(mx,my);
            if(pstate.dragJ>=0){
                if(propagateMode){const ref=_bodyKpsForFrame(state.frame)[pstate.dragJ]||[0,0];pstate._dragRef=[ref[0],ref[1]];const[s,e]=_propRange();pstate._dragRange=[s,e];const ov=parsePoseOverrides(node),entry=_ensureRangeEntry(ov,s,e),cur=entry.delta[String(pstate.dragJ)];pstate._dragBase=Array.isArray(cur)&&cur.length===2?[Number(cur[0])||0,Number(cur[1])||0]:[0,0];}
                ev.stopPropagation();ev.preventDefault();render();}
        }
    });

    cvs.addEventListener("mousemove",ev=>{
        const[mx,my]=clientToCanvas(ev);
        if(gstate.dragEye){
            const{yaw,pitch}=canvasToGaze(mx,my,gstate.dragEye),ov=parseGazeOverrides(node),key=String(state.frame);
            if(!ov.frames[key])ov.frames[key]={};
            ov.frames[key][gstate.dragEye]=[+yaw.toFixed(5),+pitch.toFixed(5)];
            writeGazeOverrides(node,ov);render();if(activeTab==="gaze")_drawGimbals();ev.stopPropagation();return;
        }
        if(activeTab==="face"&&state.dragLm>=0){
            const[xn,yn]=canvasToNorm(mx,my,cvs.width,cvs.height),ov=parseOverrides(node);
            if(propagateMode){const[s,e]=state._dragRange||_propRange(),ref=state._dragRef||[xn,yn],base=state._dragBase||[0,0],entry=_ensureRangeEntry(ov,s,e);entry.delta[String(state.dragLm)]=[(base[0]+(xn-ref[0])).toFixed(4)*1,(base[1]+(yn-ref[1])).toFixed(4)*1];writeOverrides(node,ov);render();ev.stopPropagation();return;}
            const key=String(state.frame);if(!ov.frames[key])ov.frames[key]={};ov.frames[key][String(state.dragLm)]=[+xn.toFixed(4),+yn.toFixed(4)];writeOverrides(node,ov);render();ev.stopPropagation();
        } else if(activeTab==="pose"&&pstate.dragJ>=0){
            const[xn,yn]=_poseCanvasToNorm(mx,my),ov=parsePoseOverrides(node);
            if(propagateMode){const[s,e]=pstate._dragRange||_propRange(),ref=pstate._dragRef||[xn,yn],base=pstate._dragBase||[0,0],entry=_ensureRangeEntry(ov,s,e);entry.delta[String(pstate.dragJ)]=[(base[0]+(xn-ref[0])).toFixed(5)*1,(base[1]+(yn-ref[1])).toFixed(5)*1];writePoseOverrides(node,ov);render();ev.stopPropagation();return;}
            const key=String(state.frame);if(!ov.frames[key])ov.frames[key]={};ov.frames[key][String(pstate.dragJ)]=[+xn.toFixed(5),+yn.toFixed(5)];writePoseOverrides(node,ov);render();ev.stopPropagation();
        } else {
            if(activeTab==="face"||activeTab==="gaze"){
                const newEye=pickGazeHandle(mx,my),h=newEye?-1:pickLandmark(mx,my);
                if(newEye!==gstate.hoverEye||h!==state.hoverLm){gstate.hoverEye=newEye;state.hoverLm=h;render();}
            } else if(activeTab==="pose"){
                const h=_pickJoint(mx,my);if(h!==pstate.hoverJ){pstate.hoverJ=h;render();}
            }
        }
    });

    const _endDrag=()=>{let dirty=false;if(state.dragLm>=0){state.dragLm=-1;dirty=true;}if(gstate.dragEye){gstate.dragEye=null;dirty=true;}if(pstate.dragJ>=0){pstate.dragJ=-1;dirty=true;}if(dirty)render();};
    cvs.addEventListener("mouseup",_endDrag);
    cvs.addEventListener("mouseleave",()=>{state.hoverLm=-1;gstate.hoverEye=null;pstate.hoverJ=-1;_endDrag();render();});
    cvs.addEventListener("contextmenu",ev=>{
        const[mx,my]=clientToCanvas(ev);
        if(activeTab==="face"||activeTab==="gaze"){const eye=pickGazeHandle(mx,my);if(eye){ev.preventDefault();const ov=parseGazeOverrides(node),key=String(state.frame);if(ov.frames?.[key]){delete ov.frames[key][eye];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writeGazeOverrides(node,ov);render();if(activeTab==="gaze")_drawGimbals();}}}
        if(activeTab==="pose"){ev.preventDefault();const j=_pickJoint(mx,my);if(j<0)return;const ov=parsePoseOverrides(node),key=String(state.frame);if(ov.frames?.[key]){delete ov.frames[key][String(j)];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writePoseOverrides(node,ov);render();}}
    });

    // ── Frame navigation ────────────────────────────────────────────
    function _onFrameChanged(){if(activeTab==="expr")_refreshSliders();if(activeTab==="gaze")_drawGimbals();try{_fc3dEditor?.refresh?.();}catch(_){}}
    function _stepFrame(delta){const max=parseInt(slider.max,10)||0,next=Math.min(max,Math.max(0,state.frame+delta));if(next===state.frame)return;state.frame=next;slider.value=String(next);frameLbl.textContent=`f ${next} / ${max}`;frameHint.textContent=frameLbl.textContent;render();_onFrameChanged();try{_persistSave?.();}catch(_){}}
    function _gotoFrame(idx){const max=parseInt(slider.max,10)||0,next=Math.min(max,Math.max(0,idx));state.frame=next;slider.value=String(next);frameLbl.textContent=`f ${next} / ${max}`;frameHint.textContent=frameLbl.textContent;render();_onFrameChanged();try{_persistSave?.();}catch(_){}}
    slider.addEventListener("input",()=>{state.frame=parseInt(slider.value,10)||0;frameLbl.textContent=`f ${state.frame} / ${slider.max}`;frameHint.textContent=frameLbl.textContent;render();_onFrameChanged();try{_persistSave?.();}catch(_){}});
    btnPrev.addEventListener("click",()=>_stepFrame(-1));
    btnNext.addEventListener("click",()=>_stepFrame(1));
    btnReset.addEventListener("click",()=>{
        const ov=parseOverrides(node);delete ov.frames?.[String(state.frame)];writeOverrides(node,ov);
        const pov=parsePoseOverrides(node);if(pov.frames?.[String(state.frame)]){delete pov.frames[String(state.frame)];writePoseOverrides(node,pov);}
        const gov=parseGazeOverrides(node);if(gov.frames?.[String(state.frame)]){delete gov.frames[String(state.frame)];writeGazeOverrides(node,gov);}
        const exprData=_parseCoeffsJson();if(exprData.frames?.[String(state.frame)]){delete exprData.frames[String(state.frame)];_writeCoeffsJson(exprData);}
        render();if(activeTab==="expr")_refreshSliders();if(activeTab==="gaze")_drawGimbals();
    });

    // ── Keyboard shortcuts ──────────────────────────────────────────
    cvs.addEventListener("keydown",ev=>{
        if(ev.target&&ev.target!==cvs&&(ev.target.tagName==="INPUT"||ev.target.tagName==="TEXTAREA"||ev.target.tagName==="SELECT"))return;
        if((ev.ctrlKey||ev.metaKey)&&ev.key==="z"){if(ev.shiftKey)_redo();else _undo();ev.preventDefault();return;}
        if((ev.ctrlKey||ev.metaKey)&&ev.key==="y"){_redo();ev.preventDefault();return;}
        const big=ev.shiftKey?10:1;
        switch(ev.key){
            case "ArrowLeft":case "ArrowDown":_stepFrame(-big);ev.preventDefault();break;
            case "ArrowRight":case "ArrowUp":_stepFrame(+big);ev.preventDefault();break;
            case "Home":_gotoFrame(0);ev.preventDefault();break;
            case "End":_gotoFrame(parseInt(slider.max,10)||0);ev.preventDefault();break;
            case "r":case "R":btnReset.click();ev.preventDefault();break;
            case "Escape":cvs.blur();ev.preventDefault();break;
            case "Delete":case "Backspace":
                if(activeTab==="face"&&state.hoverLm>=0){const ov=parseOverrides(node),key=String(state.frame);if(ov.frames?.[key]){delete ov.frames[key][String(state.hoverLm)];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writeOverrides(node,ov);render();}ev.preventDefault();}
                break;
            case "1":switchTab("face");ev.preventDefault();break;
            case "2":switchTab("expr");ev.preventDefault();break;
            case "3":switchTab("gaze");ev.preventDefault();break;
            case "4":switchTab("pose");ev.preventDefault();break;
            case "5":switchTab("set");ev.preventDefault();break;
        }
    });

    // ── 3D Editor lazy-load ─────────────────────────────────────────
    btn3D.addEventListener("click",async()=>{
        if(_fc3dEditor){try{_fc3dEditor.destroy();}catch(_){}_fc3dEditor=null;if(_fc3dHost){try{_fc3dHost.remove();}catch(_){}_fc3dHost=null;}btn3D.style.background=C.border;try{_persistSave?.();}catch(_){}return;}
        btn3D.style.background=C.accent;
        _fc3dHost=_el("div","margin-top:4px;");ctxPose.appendChild(_fc3dHost);
        const _wRead=(name)=>Number(readParam(node,name))||0;
        const _wWrite=(name,v)=>{let c=Number(v);if(!Number.isFinite(c))return;const opts=FC3D_PARAM_OPTS[name]||{};if(opts.min!==undefined)c=Math.max(opts.min,c);if(opts.max!==undefined)c=Math.min(opts.max,c);writeParam(node,name,c);node.setDirtyCanvas?.(true,true);};
        try{
            const mod=await import("./face_3d_editor.js");if(!_fc3dHost)return;
            _fc3dEditor=await mod.mount3DEditor(_fc3dHost,{
                theme:C,
                getLandmarks:()=>{try{return landmarksForFrame(state.frame);}catch(_){return null;}},
                getHeadPose:()=>({yaw:_wRead("head_yaw_deg"),pitch:_wRead("head_pitch_deg"),roll:_wRead("head_roll_deg"),tx:_wRead("head_tx"),ty:_wRead("head_ty"),tz:_wRead("head_tz"),scale:_wRead("head_scale"),jaw:_wRead("jaw_rot_deg"),neck_yaw:_wRead("neck_yaw_deg"),neck_pitch:_wRead("neck_pitch_deg")}),
                setHeadPose:partial=>{if(partial.yaw!==undefined)_wWrite("head_yaw_deg",partial.yaw);if(partial.pitch!==undefined)_wWrite("head_pitch_deg",partial.pitch);if(partial.roll!==undefined)_wWrite("head_roll_deg",partial.roll);if(partial.tx!==undefined)_wWrite("head_tx",partial.tx);if(partial.ty!==undefined)_wWrite("head_ty",partial.ty);if(partial.tz!==undefined)_wWrite("head_tz",partial.tz);if(partial.scale!==undefined)_wWrite("head_scale",partial.scale);if(partial.jaw!==undefined)_wWrite("jaw_rot_deg",partial.jaw);if(partial.neck_yaw!==undefined)_wWrite("neck_yaw_deg",partial.neck_yaw);if(partial.neck_pitch!==undefined)_wWrite("neck_pitch_deg",partial.neck_pitch);try{_scheduleLocalMirror();_scheduleServerResync();}catch(_){}},
                onClose:()=>{_fc3dEditor=null;if(_fc3dHost){try{_fc3dHost.remove();}catch(_){}_fc3dHost=null;}btn3D.style.background=C.border;try{_persistSave?.();}catch(_){}},
            });
            try{_persistSave?.();}catch(_){}
        }catch(err){if(_fc3dHost){_fc3dHost.textContent="3D editor unavailable: "+(err?.message||String(err));_fc3dHost.style.cssText=`margin-top:4px;padding:4px 6px;color:#ff7070;background:${C.canvas_bg};border:1px solid ${C.border};border-radius:3px;font:10px ui-monospace,monospace;`;}btn3D.style.background=C.border;}
    });

    // ── Live-preview engine ─────────────────────────────────────────
    let _localBaseline=null,_mirrorRaf=0,_resyncTimer=0,_resyncInFlight=false,_resyncSeq=0;
    const DOF_NAMES=["head_yaw_deg","head_pitch_deg","head_roll_deg","head_tx","head_ty","head_tz","head_scale","jaw_rot_deg","neck_yaw_deg","neck_pitch_deg","gaze_yaw_deg","gaze_pitch_deg","expression_strength","expression_clamp","blend_strength"];
    function _readAllDOF() {
        const out = {};
        for (const k of DOF_NAMES) out[k] = Number(readParam(node, k)) || 0;
        for (const k of [
            "expression_coeffs_json", "head_pose_json", "gaze_json",
            "use_metas", "propagate_expression", "propagate_head", "propagate_gaze",
            "landmark_overrides_json", "pose_overrides_json", "gaze_overrides_json",
            "frame_start", "frame_end",
        ]) out[k] = readParam(node, k);
        for (const k of ["blend_mouth", "blend_brows", "blend_eyes", "blend_jaw"]) {
            out[k] = !!readParam(node, k);
        }
        return out;
    }
    function _deepCloneFrames(frames){if(!Array.isArray(frames))return[];return frames.map(f=>{if(!f)return f;const c={...f};if(Array.isArray(f.kps))c.kps=f.kps.map(p=>Array.isArray(p)?p.slice():p);return c;});}
    function _captureBaseline(){_localBaseline={faceFrames:Array.isArray(state.meta?.frames)?state.meta.frames.map(fr=>fr?{...fr}:fr):[],poseFrames:_deepCloneFrames(pstate.frames),faceLm:{},widgetSnap:_readAllDOF()};const lmsF=state.meta?.frames;if(Array.isArray(lmsF)){for(let i=0;i<lmsF.length;i++){try{const lms=landmarksForFrame(i);if(Array.isArray(lms)&&lms.length===68)_localBaseline.faceLm[i]=lms.map(p=>p.slice());}catch(_){}}}}
    function _mirrorTransform(frameIdx){
        if(!_localBaseline)return;const base=_localBaseline.widgetSnap,now=_readAllDOF();
        const dYaw=(now.head_yaw_deg-base.head_yaw_deg)*Math.PI/180,dPitch=(now.head_pitch_deg-base.head_pitch_deg)*Math.PI/180,dRoll=(now.head_roll_deg-base.head_roll_deg)*Math.PI/180;
        const dTx=now.head_tx-base.head_tx,dTy=now.head_ty-base.head_ty,dTz=now.head_tz-base.head_tz;
        const rScale=(now.head_scale||1)/(base.head_scale||1),dNYaw=(now.neck_yaw_deg-base.neck_yaw_deg)*Math.PI/180,dNPit=(now.neck_pitch_deg-base.neck_pitch_deg)*Math.PI/180;
        const cosR=Math.cos(dRoll),sinR=Math.sin(dRoll),cosY=Math.cos(dYaw),cosP=Math.cos(dPitch);
        const tzClamp=Math.max(-0.75,Math.min(3,dTz)),zoomTz=1/(1+tzClamp);
        const baseLm=_localBaseline.faceLm[frameIdx],faceFr=state.meta?.frames?.[frameIdx];
        if(baseLm&&faceFr){let cx=0,cy=0;for(const p of baseLm){cx+=p[0];cy+=p[1];}cx/=baseLm.length;cy/=baseLm.length;const s=rScale*zoomTz;let mnX=1,mxX=0;for(const p of baseLm){if(p[0]<mnX)mnX=p[0];if(p[0]>mxX)mxX=p[0];}const bbW=Math.max(1e-6,mxX-mnX);const dNorm=new Array(68),newLm=new Array(68);for(let i=0;i<baseLm.length;i++){let x=baseLm[i][0]-cx,y=baseLm[i][1]-cy;x*=s;y*=s;const rx=x*cosR-y*sinR,ry=x*sinR+y*cosR;let nx=rx*cosY+cx+dTx*bbW,ny=ry*cosP+cy+dTy*bbW;newLm[i]=[nx,ny];dNorm[i]=[(nx-baseLm[i][0])/bbW,(ny-baseLm[i][1])/bbW];}faceFr.lms=newLm;faceFr.d_norm=dNorm;}
        const basePose=_localBaseline.poseFrames?.[frameIdx],livePose=pstate.frames?.[frameIdx];
        if(basePose&&livePose&&Array.isArray(basePose.kps)&&Array.isArray(livePose.kps)&&livePose.kps.length===18){const neck=basePose.kps[1],baseKps=basePose.kps;if(Array.isArray(neck)&&neck.length>=2&&(Math.abs(dNYaw)>1e-3||Math.abs(dNPit)>1e-3)){const nx=neck[0],ny=neck[1],headCluster=[0,14,15,16,17],cyN=Math.cos(dNYaw),cpN=Math.cos(dNPit),spN=Math.sin(dNPit);const newKps=baseKps.map(p=>Array.isArray(p)?p.slice():p);for(const idx of headCluster){const p=baseKps[idx];if(!Array.isArray(p))continue;newKps[idx][0]=nx+(p[0]-nx)*cyN;newKps[idx][1]=ny+(p[1]-ny)*cpN-spN*Math.abs(p[1]-ny);}livePose.kps=newKps;}else livePose.kps=baseKps.map(p=>Array.isArray(p)?p.slice():p);}
    }
    function _scheduleLocalMirror(){if(_mirrorRaf)return;_mirrorRaf=requestAnimationFrame(()=>{_mirrorRaf=0;try{_mirrorTransform(state.frame);}catch(_){}try{render();}catch(_){}});}
    async function _serverResync(){const seq=++_resyncSeq;if(_resyncInFlight)return;_resyncInFlight=true;try{const body={node_id:String(node.id),frame_idx:state.frame,..._readAllDOF()};const resp=await fetch("/c2c/fc3d_preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(seq!==_resyncSeq)return;if(resp.status===412){frameHint.textContent="queue node once for server preview";return;}if(!resp.ok)return;const data=await resp.json();if(seq!==_resyncSeq)return;const fIdx=data.frame_idx,faceArr=data.face_norm,bodyArr=data.body_kps;if(Array.isArray(faceArr)&&_localBaseline?.faceLm?.[fIdx]){const baseLm=_localBaseline.faceLm[fIdx];let mnX=1,mxX=0,mnY=1,mxY=0;for(const p of baseLm){if(p[0]<mnX)mnX=p[0];if(p[0]>mxX)mxX=p[0];if(p[1]<mnY)mnY=p[1];if(p[1]>mxY)mxY=p[1];}const bbW=Math.max(1e-6,mxX-mnX),bbH=Math.max(1e-6,mxY-mnY);const absLm=faceArr.map(p=>[mnX+p[0]*bbW,mnY+p[1]*bbH]);const faceFr=state.meta?.frames?.[fIdx];if(faceFr){faceFr.lms=absLm;faceFr.d_norm=absLm.map((p,i)=>[(p[0]-baseLm[i][0])/bbW,(p[1]-baseLm[i][1])/bbH]);}}if(Array.isArray(bodyArr)&&pstate.frames?.[fIdx]){pstate.frames[fIdx].kps=bodyArr.map(p=>Array.isArray(p)?p.slice():[NaN,NaN]);}try{render();}catch(_){}}catch(_){}finally{_resyncInFlight=false;}}
    function _scheduleServerResync(){if(_resyncTimer)clearTimeout(_resyncTimer);_resyncTimer=setTimeout(()=>{_resyncTimer=0;_serverResync();},120);}
    function _hookConfigWidget() {
        const w = readWidget(node, CONFIG_WIDGET);
        if (!w || w.__fc3d_hooked) return;
        const orig = w.callback;
        w.callback = function (v) {
            try { orig?.call(this, v, node, w); } catch (_) {}
            try { _syncSettingsFromWidgets(); } catch (_) {}
            _scheduleLocalMirror();
            _scheduleServerResync();
        };
        w.__fc3d_hooked = true;
    }
    setTimeout(_hookConfigWidget, 0);
    setTimeout(_hookConfigWidget, 250);
    _onWidgetChg = () => { try { _scheduleLocalMirror(); _scheduleServerResync(); } catch (_) {} };

    // ── Persistence ─────────────────────────────────────────────────
    const _persistKey=()=>`mec.fc3d.overlay.${node.id??"_"}`;
    let _persistRaf=0;
    function _persistSave(){if(_persistRaf)return;_persistRaf=requestAnimationFrame(()=>{_persistRaf=0;try{localStorage.setItem(_persistKey(),JSON.stringify({tab:activeTab,frame:state.frame,editor3d:!!_fc3dEditor}));}catch(_){}});}
    function _persistLoad(){try{const raw=localStorage.getItem(_persistKey());return raw?JSON.parse(raw):null;}catch(_){return null;}}

    // ── Public API ──────────────────────────────────────────────────
    const api = {
        root,
        update(meta) {
            state.meta = meta;
            if (meta?.frames?.some?.((fr) => fr?.lms?.length >= 68)) {
                state.meta._synthetic = false;
            }
            const pmeta=(meta&&typeof meta==="object")?meta.pose:null;
            if(pmeta&&Array.isArray(pmeta.frames)){pstate.frames=pmeta.frames;if(Array.isArray(pmeta.edges)&&pmeta.edges.length)pstate.edges=pmeta.edges;if(Array.isArray(pmeta.joint_names)&&pmeta.joint_names.length)pstate.names=pmeta.joint_names;}else pstate.frames=[];
            const nFace=Array.isArray(meta?.frames)?meta.frames.length:0,nPose=pstate.frames.length,n=Math.max(nFace,nPose);
            slider.max=String(Math.max(0,n-1));
            if(state.frame>Number(slider.max)){state.frame=Number(slider.max);slider.value=String(state.frame);}
            frameLbl.textContent=`f ${state.frame} / ${slider.max}`;
            frameHint.textContent=frameLbl.textContent;
            _captureBaseline();render();drawTimeline();
        },
        focus(){try{cvs.focus();}catch(_){}},
        resetFrame(){try{btnReset.click();}catch(_){}},
        gotoFirst(){_gotoFrame(0);},
        gotoLast(){_gotoFrame(parseInt(slider.max,10)||0);},
        step(delta){_stepFrame(delta);},
        setView(v){const map={face:"face",pose:"pose",both:"face"};switchTab(map[v]||v);},
        mirrorRefresh(){try{_scheduleLocalMirror();}catch(_){}},
        serverResync(){try{_scheduleServerResync();}catch(_){}},
        resetTransform(){for(const k of DOF_NAMES){let def=0;if(k==="head_scale"||k==="expression_strength"||k==="blend_strength")def=1;if(k==="expression_clamp")def=FC3D_DEFAULT_CFG.expression_clamp;writeParam(node,k,def);}node.setDirtyCanvas?.(true,true);_scheduleLocalMirror();_scheduleServerResync();},
        setVideoFrames(urls){try{_setVideoFrames(urls);render();}catch(_){}},
        undo(){_undo();},redo(){_redo();},
        _scheduleUndoSnap(){_scheduleUndoSnap();},
        _relayout,
        _scheduleRelayout,
        _syncSettingsFromWidgets,
        getHeight: () => _editorH,
        onNodeResize(nodeW, nodeH) {
            const applied = _fillAvailableHeight(nodeW, nodeH);
            _lastLayoutH = applied;
            try { render(); drawTimeline(); } catch (_) {}
            return applied;
        },
        destroy3D(){
            try { _resizeObs?.disconnect(); } catch (_) {}
            _resizeObs = null;
            try { _fc3dEditor?.destroy(); } catch (_) {}
            _fc3dEditor = null;
            if (_fc3dHost) { try { _fc3dHost.remove(); } catch (_) {} _fc3dHost = null; }
            _videoFrames = [];
            if (root._gazeCleanups) {
                root._gazeCleanups.forEach(fn => { try { fn(); } catch (_) {} });
                root._gazeCleanups = [];
            }
        },
    };

    // ── Initial state restore + first draw ──────────────────────────
    const _saved=_persistLoad();
    if(_saved?.tab&&["face","expr","gaze","pose","set"].includes(_saved.tab))switchTab(_saved.tab);
    if(_saved&&Number.isFinite(_saved.frame)){state.frame=Math.max(0,Number(_saved.frame)|0);try{slider.value=String(state.frame);}catch(_){}frameLbl.textContent=`f ${state.frame} / ${slider.max}`;frameHint.textContent=frameLbl.textContent;}
    if(_saved?.editor3d)setTimeout(()=>{try{btn3D.click();}catch(_){}},0);
    render(); drawTimeline();
    _syncSettingsFromWidgets();
    _relayout();
    node._faceOverlay = api;
    return api;
}

// ═══════════════════════════════════════════════════════════════════
//  Extension registration
// ═══════════════════════════════════════════════════════════════════
try { document.getElementById("fc3d-face-controller-css")?.remove(); } catch (_) {}

/** Build DOM editor + size hooks (idempotent). */
function _fc3dSetupNode(node) {
    if (!node || node._faceOverlay) return;

    for (const wg of node.widgets || []) {
        if (wg.options === undefined) wg.options = {};
    }
    _fc3dMigrateLegacyWidgets(node);

    const overlay = buildEditor(node);
    node._faceOverlay = overlay;

    const domW = node.addDOMWidget("face_overlay", "canvas", overlay.root, {
        getValue() { return ""; },
        setValue() {},
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => FC3D_MIN_H,
        getHeight: () => _fc3dGetContentHeight(node),
        getMaxHeight: () => FC3D_MAX_H,
    });
    _fc3dSetupDomWidget(node, domW);

    const hideChrome = () => _fc3dHideChromeWidgets(node, domW);

    node._fc3dDomW = domW;
    node._fc3dHideChrome = hideChrome;
    node.resizable = true;
    node.minimum_size = [FC3D_NODE_W, FC3D_MIN_H];

    hideChrome();

    if (node._cachedOverlayMeta) {
        try { overlay.update(JSON.parse(node._cachedOverlayMeta)); } catch (_) {}
    }

    // Single deferred sync — no clamp loop, no repeated setTimeouts
    const _syncOnce = () => {
        hideChrome();
        _fc3dSyncNodeSize(node);
    };
    setTimeout(_syncOnce, 0);
    setTimeout(_syncOnce, 300);

    // Simple onResize — just notify the editor to relayout, no size overrides
    if (!node._fc3dHooksBound) {
        node._fc3dHooksBound = true;
        const _origOnAdded = node.onAdded;
        node.onAdded = function () {
            _origOnAdded?.apply(this, arguments);
            _syncOnce();
        };

        const _origOnResize = node.onResize;
        node.onResize = function (size) {
            _origOnResize?.apply(this, arguments);
            const sz = size || node.size;
            const nw = Math.max(FC3D_NODE_W, sz?.[0] || FC3D_NODE_W);
            const nh = Math.max(FC3D_MIN_H, Math.min(FC3D_MAX_H, sz?.[1] || FC3D_MIN_H));
            _fc3dWriteNodeSize(node, nw, nh);
            try { overlay.onNodeResize?.(nw, nh); } catch (_) {}
            try { overlay._relayout?.(); } catch (_) {}
            requestAnimationFrame(() => {
                try { overlay.render?.(); overlay.drawTimeline?.(); } catch (_) {}
            });
        };
    }

    _fc3dInstances.add(node);
}

app.registerExtension({
    name: "MEC.WanFaceController3DV2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const _origIsWidgetVisible = nodeType.prototype.isWidgetVisible;
        nodeType.prototype.isWidgetVisible = function (widget) {
            if (widget?.name === "face_overlay") return true;
            if (widget?.__fc3d_chrome_hidden) return false;
            if (widget?.options?.hidden) return false;
            if (widget?.type === "hidden") return false;
            return _origIsWidgetVisible ? _origIsWidgetVisible.call(this, widget) : true;
        };

        const _origProtoCS = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (outW) {
            const w = Math.max(FC3D_NODE_W, this.size?.[0] || FC3D_NODE_W);
            const cap = FC3D_MAX_H;
            if (_origProtoCS) {
                const sz = _origProtoCS.call(this, outW);
                return [Math.max(sz[0] || w, FC3D_NODE_W), Math.min(sz[1] || cap, cap)];
            }
            const h = Math.min(Math.max(this.size?.[1] || FC3D_MIN_H, FC3D_MIN_H), cap);
            return [w, h];
        };

        // Hide backend-only widgets the moment LiteGraph adds them — prevents
        // multiline STRING fields from reserving height before onNodeCreated runs.
        const _origAddWidget = nodeType.prototype.addWidget;
        if (_origAddWidget) {
            nodeType.prototype.addWidget = function (...args) {
                const w = _origAddWidget.apply(this, args);
                if (w?.name && w.name !== "face_overlay") {
                    _fc3dHideChromeWidget(w);
                }
                return w;
            };
        }

        // Multiline STRING inputs (fc3d_config_json etc) go through addDOMWidget
        const _origAddDOMWidget = nodeType.prototype.addDOMWidget;
        if (_origAddDOMWidget) {
            nodeType.prototype.addDOMWidget = function (...args) {
                const w = _origAddDOMWidget.apply(this, args);
                if (w?.name && w.name !== "face_overlay") {
                    _fc3dHideChromeWidget(w);
                }
                return w;
            };
        }

        // Some paths use addCustomWidget
        const _origAddCustomWidget = nodeType.prototype.addCustomWidget;
        if (_origAddCustomWidget) {
            nodeType.prototype.addCustomWidget = function (w) {
                const result = _origAddCustomWidget.call(this, w);
                if (w?.name && w.name !== "face_overlay") {
                    _fc3dHideChromeWidget(w);
                }
                return result;
            };
        }

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            _fc3dSetupNode(this);
        };

        const _configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            _configure?.apply(this, arguments);
            try {
                _fc3dMigrateLegacyWidgets(this);
                this._fc3dHideChrome?.();
                this._faceOverlay?._syncSettingsFromWidgets?.();
                const contentH = this._faceOverlay?.getHeight?.() || FC3D_MIN_H;
                const savedH = this.size?.[1];
                if (savedH > contentH + 80) {
                    _fc3dWriteNodeSize(this, Math.max(FC3D_NODE_W, this.size?.[0] || FC3D_NODE_W), contentH);
                }
                _fc3dSyncNodeSize(this);
                this._faceOverlay?._scheduleRelayout?.();
            } catch (_) {}
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
                try { this._fc3dHideChrome?.(); this._faceOverlay?._scheduleRelayout?.(); } catch (_) {}
            }
        };

        const _removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._faceOverlay?.destroy3D?.(); } catch (_) {}
            _fc3dInstances.delete(this);
            return _removed?.apply(this, arguments);
        };

        try { window.__FC3D_NODE_DEF_READY__ = true; } catch (_) {}
    },

    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        requestAnimationFrame(() => {
            try {
                if (!node._faceOverlay) _fc3dSetupNode(node);
                node._fc3dHideChrome?.();
                _fc3dSyncNodeSize(node);
            } catch (_) {}
        });
    },

    loadedGraphNode(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        requestAnimationFrame(() => {
            try {
                if (!node._faceOverlay) _fc3dSetupNode(node);
                node._fc3dHideChrome?.();
                _fc3dSyncNodeSize(node);
            } catch (_) {}
        });
    },
});

try { window.__FC3D_EXT__ = true; } catch (_) {}

// ── Command palette ─────────────────────────────────────────────────
const _fc3dInstances = new Set();
function _fc3dActive() { let last = null; for (const n of _fc3dInstances) last = n; return last?._faceOverlay || null; }
function _fc3dRegisterActions() {
    const reg = window.__C2C_ACTIONS__?.register;
    if (typeof reg !== "function") return;
    const enabled = () => _fc3dInstances.size > 0;
    const actions = [
        { id:"mec.faceController.focus", title:"Face Controller: Focus canvas", icon:"\u25CE", keywords:["face","focus"], run:()=>_fc3dActive()?.focus() },
        { id:"mec.faceController.resetFrame", title:"Face Controller: Reset frame", icon:"\u27F2", keywords:["face","reset"], run:()=>_fc3dActive()?.resetFrame() },
        { id:"mec.faceController.gotoFirst", title:"Face Controller: First frame", icon:"\u23EE", keywords:["face","first"], run:()=>_fc3dActive()?.gotoFirst() },
        { id:"mec.faceController.gotoLast", title:"Face Controller: Last frame", icon:"\u23ED", keywords:["face","last"], run:()=>_fc3dActive()?.gotoLast() },
        { id:"mec.faceController.viewFace", title:"Face Controller: Face tab", icon:"\uD83D\uDC64", keywords:["face","tab"], run:()=>_fc3dActive()?.setView("face") },
        { id:"mec.faceController.viewPose", title:"Face Controller: Pose tab", icon:"\uD83E\uDDB4", keywords:["pose","tab"], run:()=>_fc3dActive()?.setView("pose") },
        { id:"mec.faceController.mirrorRefresh", title:"Face Controller: Refresh mirror", icon:"\u21BB", keywords:["mirror","preview"], run:()=>_fc3dActive()?.mirrorRefresh() },
        { id:"mec.faceController.serverResync", title:"Face Controller: Server resync", icon:"\u21C5", keywords:["sync","server"], run:()=>_fc3dActive()?.serverResync() },
        { id:"mec.faceController.resetTransform", title:"Face Controller: Reset DOFs", icon:"\u2316", keywords:["reset","transform"], run:()=>_fc3dActive()?.resetTransform() },
        { id:"mec.faceController.undo", title:"Face Controller: Undo", icon:"\u21B6", keywords:["undo"], run:()=>_fc3dActive()?.undo() },
        { id:"mec.faceController.redo", title:"Face Controller: Redo", icon:"\u21B7", keywords:["redo"], run:()=>_fc3dActive()?.redo() },
    ];
    for (const a of actions) { try { reg({ ...a, kind:"command", scope:"graph", enabled }); } catch (_) {} }
}
setTimeout(_fc3dRegisterActions, 0);
setTimeout(_fc3dRegisterActions, 1000);
