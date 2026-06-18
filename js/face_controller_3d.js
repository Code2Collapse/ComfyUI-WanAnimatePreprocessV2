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
// Neutral front-facing OpenPose-18 skeleton in normalized [0,1] coords, so the
// Pose tab shows a manipulable default figure BEFORE any input is queued (the
// Face tab already had a synthetic default via makeDefaultMeta; the Pose tab did
// not, which is why it read "no pose data — queue once" and the pitch/yaw/roll
// gimbal had no coords to move). Order matches POSE18_NAMES.
const DEFAULT_POSE18_KPS = [
    [0.50,0.12],            // 0  nose
    [0.50,0.19],            // 1  neck
    [0.41,0.21],[0.37,0.35],[0.35,0.48],   // 2-4  r shoulder/elbow/wrist
    [0.59,0.21],[0.63,0.35],[0.65,0.48],   // 5-7  l shoulder/elbow/wrist
    [0.45,0.52],[0.44,0.72],[0.44,0.94],   // 8-10 r hip/knee/ankle
    [0.55,0.52],[0.56,0.72],[0.56,0.94],   // 11-13 l hip/knee/ankle
    [0.47,0.105],[0.53,0.105],             // 14-15 r/l eye
    [0.44,0.12],[0.56,0.12],               // 16-17 r/l ear
];
function _defaultPoseFrame() {
    return { i: 0, ok: true, w: 512, h: 768, kps: DEFAULT_POSE18_KPS.map(p => [p[0], p[1]]) };
}
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
    // \u2500\u2500 full FACS facial set (user's AU grid) \u2500\u2500
    {id:"eye_wide",label:"Upper Lid Raise",au:"AU5",icon:"\u25C9"},
    {id:"lid_tighten",label:"Lid Tightener",au:"AU7",icon:"\u2501"},
    {id:"upper_lip_raise",label:"Upper Lip Raise",au:"AU10",icon:"\u2934"},
    {id:"nasolabial",label:"Nasolabial Deepen",au:"AU11",icon:"\u2298"},
    {id:"sharp_lip_pull",label:"Sharp Lip Puller",au:"AU13",icon:"\u2323"},
    {id:"dimpler",label:"Dimpler (smirk)",au:"AU14",icon:"\u25D5"},
    {id:"lower_lip_depress",label:"Lower Lip Depress",au:"AU16",icon:"\u2935"},
    {id:"chin_raise",label:"Chin Raiser",au:"AU17",icon:"\u2934"},
    {id:"lip_stretch",label:"Lip Stretcher",au:"AU20",icon:"\u2194"},
    {id:"lip_funnel",label:"Lip Funneler",au:"AU22",icon:"\u25CC"},
    {id:"lip_tighten",label:"Lip Tightener",au:"AU23",icon:"\u2501"},
    {id:"lip_press",label:"Lip Pressor",au:"AU24",icon:"\u2550"},
    {id:"lip_suck",label:"Lip Suck",au:"AU28",icon:"\u25D6"},
    {id:"lid_droop",label:"Lid Droop",au:"AU41",icon:"\u25E1"},
    {id:"squint",label:"Squint",au:"AU44",icon:"\u25E0"},
];
const EMOTION_PRESETS = {
    "Neutral":{},
    "Happy":{smile:0.8,cheek_raise:0.6,eye_close_L:0.15,eye_close_R:0.15},
    "Sad":{frown:0.7,brow_inner_raise:0.5,brow_furrow:0.3},
    "Angry":{brow_furrow:0.9,nose_wrinkle:0.5,lip_pucker:0.3,frown:0.4},
    "Surprise":{brow_inner_raise:0.8,brow_outer_raise:0.7,mouth_open:0.8,jaw_drop:0.5},
    "Fear":{brow_inner_raise:0.9,brow_outer_raise:0.4,mouth_open:0.5,eye_close_L:-0.2,eye_close_R:-0.2},
    "Disgust":{nose_wrinkle:0.8,brow_furrow:0.4,frown:0.5,cheek_raise:0.3,lip_pucker:0.4},
    // ── richer emotion library (built from the 12 FACS AUs) ──
    "Smoldering Contempt":{lid_droop:0.4,dimpler:0.42,upper_lip_raise:0.18,eye_close_L:0.15,brow_furrow:0.18,nose_wrinkle:0.1},
    "Smirk":{dimpler:0.6,smile:0.28,eye_close_L:0.2,cheek_raise:0.2},
    "Smug":{dimpler:0.4,eye_close_L:0.25,eye_close_R:0.25,brow_outer_raise:0.3,cheek_raise:0.3},
    "Skeptical":{brow_outer_raise:0.6,lid_tighten:0.3,brow_furrow:0.15,lip_pucker:0.2},
    "Sneer":{upper_lip_raise:0.6,nose_wrinkle:0.5,brow_furrow:0.25,dimpler:0.2},
    "Determined":{brow_furrow:0.6,lip_press:0.5,lid_tighten:0.2},
    "Grimace":{lip_stretch:0.7,lip_press:0.4,brow_furrow:0.3,squint:0.3},
    "Wince":{squint:0.6,lip_tighten:0.5,nose_wrinkle:0.3,brow_furrow:0.3},
    "Pained":{brow_inner_raise:0.6,squint:0.5,lip_stretch:0.4,chin_raise:0.2},
    "Shocked":{eye_wide:0.9,brow_inner_raise:0.7,brow_outer_raise:0.6,mouth_open:0.7,jaw_drop:0.4},
    "Disbelief Stare":{eye_wide:0.6,brow_outer_raise:0.5,lid_tighten:0.2,lip_press:0.2},
    "Sleepy":{eye_close_L:0.6,eye_close_R:0.6,brow_inner_raise:0.2,jaw_drop:0.15},
    "Crying":{frown:0.8,brow_inner_raise:0.85,eye_close_L:0.4,eye_close_R:0.4,mouth_open:0.3},
    "Laughing":{smile:1.0,cheek_raise:0.9,eye_close_L:0.5,eye_close_R:0.5,mouth_open:0.55,jaw_drop:0.3},
    "Ecstatic":{smile:1.0,cheek_raise:1.0,brow_outer_raise:0.4,mouth_open:0.4,eye_close_L:0.3,eye_close_R:0.3},
    "Pout":{lip_pucker:0.7,frown:0.3,brow_inner_raise:0.3},
    "Worried":{brow_inner_raise:0.7,brow_furrow:0.4,frown:0.3,lip_pucker:0.2},
    "Confused":{brow_outer_raise:0.5,brow_furrow:0.3,lip_pucker:0.25,eye_close_L:0.15},
    "Suspicious":{eye_close_L:0.45,eye_close_R:0.28,brow_furrow:0.35,lip_pucker:0.2},
    "Bored":{eye_close_L:0.4,eye_close_R:0.4,frown:0.2,brow_outer_raise:-0.2},
    "Disbelief":{brow_outer_raise:0.7,brow_inner_raise:0.5,mouth_open:0.4,jaw_drop:0.2},
    "Wink L":{eye_close_L:1.0,smile:0.4,cheek_raise:0.3},
    "Wink R":{eye_close_R:1.0,smile:0.4,cheek_raise:0.3},
};
const PRESET_QUICK = ["Neutral", "Happy", "Sad", "Angry", "Surprise"];
const PRESET_MORE = ["Fear", "Disgust", "Smoldering Contempt", "Smirk", "Smug",
    "Skeptical", "Sneer", "Determined", "Grimace", "Wince", "Pained", "Shocked",
    "Disbelief Stare", "Sleepy", "Crying", "Laughing", "Ecstatic", "Pout",
    "Worried", "Confused", "Suspicious", "Bored", "Disbelief", "Wink L", "Wink R"];

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
    // ── full FACS facial set (mirrors expression_3d_coeffs._BASIS_TABLE) ──
    eye_wide: {37:[0,-0.012],38:[0,-0.012],43:[0,-0.012],44:[0,-0.012]},
    lid_tighten: {40:[0,-0.009],41:[0,-0.009],46:[0,-0.009],47:[0,-0.009],37:[0,0.004],38:[0,0.004],43:[0,0.004],44:[0,0.004]},
    upper_lip_raise: {50:[0,-0.018],51:[0,-0.02],52:[0,-0.018],49:[-0.004,-0.012],53:[0.004,-0.012],31:[0,-0.006],35:[0,-0.006],61:[0,-0.012],62:[0,-0.014],63:[0,-0.012]},
    nasolabial: {49:[-0.003,-0.008],53:[0.003,-0.008],31:[0,-0.004],35:[0,-0.004]},
    sharp_lip_pull: {48:[-0.03,-0.035],54:[0.03,-0.035],49:[-0.018,-0.024],53:[0.018,-0.024]},
    dimpler: {48:[0.012,-0.004],54:[-0.012,-0.004],49:[0.006,0],53:[-0.006,0]},
    lower_lip_depress: {57:[0,0.02],56:[0,0.015],58:[0,0.015],66:[0,0.014]},
    chin_raise: {57:[0,-0.018],56:[0,-0.014],58:[0,-0.014],66:[0,-0.012],8:[0,-0.01],7:[0,-0.006],9:[0,-0.006]},
    lip_stretch: {48:[-0.022,0.004],54:[0.022,0.004],49:[-0.012,0.002],53:[0.012,0.002],60:[-0.012,0],64:[0.012,0]},
    lip_funnel: {48:[0.014,0],54:[-0.014,0],51:[0,-0.004],57:[0,0.004],50:[0,-0.003],52:[0,-0.003]},
    lip_tighten: {48:[0.008,0],54:[-0.008,0],50:[0,0.003],52:[0,0.003],56:[0,-0.003],58:[0,-0.003],51:[0,0.002],57:[0,-0.002]},
    lip_press: {50:[0,0.004],51:[0,0.005],52:[0,0.004],58:[0,-0.004],57:[0,-0.005],56:[0,-0.004],62:[0,0.003],66:[0,-0.003]},
    lip_suck: {50:[0,0.005],51:[0,0.006],52:[0,0.005],56:[0,-0.005],57:[0,-0.006],58:[0,-0.005],48:[0.006,0],54:[-0.006,0]},
    lid_droop: {37:[0,0.01],38:[0,0.01],43:[0,0.01],44:[0,0.01]},
    squint: {40:[0,-0.012],41:[0,-0.012],46:[0,-0.012],47:[0,-0.012],19:[0,0.008],24:[0,0.008]},
};

const CANVAS_MIN_PX = 140;
const CANVAS_MAX_PX = 700;       // headroom for big monitors; safe now that growth
                                 // only happens via the explicit proportional grip drag
const FC3D_CANVAS_VIEW_PX = 200;
const FC3D_NODE_W = 300;
const FC3D_MIN_H = 240;          // let the node be made compact
const FC3D_MAX_H = 1400;         // was 680 — let users drag the node taller without snap-back
const FC3D_CTX_SCROLL_MAX = 340;
/** Per-tab canvas height caps — keeps the default node compact. */
const FC3D_TAB_CANVAS_PX = { face: 200, expr: 190, gaze: 180, pose: 200, set: 170 };
// Per-tab MAX height of the scrollable options strip. Only bites when a panel
// is EXPANDED (collapsed panels measure ~56px, so the default node stays
// compact); raised so expanded options (12 AU sliders, per-AU limits, FACS
// lanes, live preview, settings) have real viewing/scroll room rather than a
// cramped ~64px window. ("Node looking smaller in option-viewing space.")
const FC3D_TAB_CTX_MAX = { face: 320, expr: 360, gaze: 220, pose: 260, set: 320 };

const FC3D_HIDE_STYLE_ID = "fc3d-wan-face-hide-css";
/** Detection / live-edit control widgets that stay VISIBLE in the node body so
 *  the user can tweak them. Everything else (JSON sync blobs etc.) is hidden as
 *  chrome since the in-canvas editor owns it. */
const FC3D_VISIBLE_WIDGETS = new Set([
    "detection_threshold", "pose_threshold", "use_clahe",
    "detect_rescale", "fallback_to_full_frame",
]);

/** Collapse chrome widgets in both LiteGraph canvas mode AND Vue mode. */
function _fc3dHideChromeWidget(w) {
    if (!w) return;
    // Keep the face editor + the detection controls visible; hide the rest.
    if (w.name === "face_overlay" || FC3D_VISIBLE_WIDGETS.has(w.name)) return;
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

/** Inject the editor "skin" — a scoped stylesheet that upgrades the look of
 *  the tabbed face editor (tab bar, buttons, transport, slider, sections,
 *  chips, canvas frame, empty-state banner). Scoped to `.fc3d-editor-root`
 *  so it never touches native ComfyUI or other nodes. All colours read the
 *  shared `--c2c-*` Catppuccin tokens with hex fallbacks. Additive only —
 *  the editor keeps working if this never loads. */
const FC3D_SKIN_STYLE_ID = "fc3d-wan-face-skin-css";
function _fc3dEnsureSkinCss() {
    if (document.getElementById(FC3D_SKIN_STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = FC3D_SKIN_STYLE_ID;
    st.textContent = `
.fc3d-editor-root {
    background: var(--c2c-bg, #1a1a22);
    border: 1px solid var(--c2c-surface0, #2a2a35);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.02);
    font: 12px ui-sans-serif, system-ui, sans-serif;
}
/* Segmented tab bar */
.fc3d-editor-root .fc3d-tabbar {
    background: var(--c2c-bg3, #0e0e16);
    border: 1px solid var(--c2c-surface0, #2a2a35);
    border-radius: 8px; padding: 3px; gap: 2px;
}
.fc3d-editor-root .fc3d-tab {
    flex: 1; border: none; border-radius: 6px; cursor: pointer;
    padding: 6px 4px; font: 600 11px ui-sans-serif, system-ui;
    color: var(--c2c-sub, #a6adc8); background: transparent;
    transition: background var(--c2c-dur-fast,120ms) var(--c2c-ease-out, cubic-bezier(.16,1,.3,1)),
                color var(--c2c-dur-fast,120ms) ease;
}
.fc3d-editor-root .fc3d-tab:hover:not(.active) {
    background: rgba(255,255,255,.05); color: var(--c2c-fg, #cdd6f4);
}
.fc3d-editor-root .fc3d-tab.active {
    background: var(--c2c-blue, #89b4fa); color: var(--c2c-bg3, #11111b);
    box-shadow: 0 1px 2px rgba(0,0,0,.3);
}
.fc3d-editor-root .fc3d-tab:focus-visible { outline: none; box-shadow: 0 0 0 2px var(--c2c-blue,#89b4fa); }
/* Buttons (additive states layered over each caller's inline colour) */
.fc3d-editor-root .fc3d-btn {
    transition: filter var(--c2c-dur-fast,120ms) ease, transform 80ms ease, box-shadow 120ms ease;
}
.fc3d-editor-root .fc3d-btn:hover:not(:disabled) { filter: brightness(1.18); }
.fc3d-editor-root .fc3d-btn:active:not(:disabled) { transform: translateY(1px); }
.fc3d-editor-root .fc3d-btn:disabled { opacity: .4; cursor: default; }
.fc3d-editor-root .fc3d-btn:focus-visible { outline: none; box-shadow: 0 0 0 2px var(--c2c-blue,#89b4fa); }
/* Transport strip */
.fc3d-editor-root .fc3d-transport {
    background: var(--c2c-bg3, #0e0e16);
    border: 1px solid var(--c2c-surface0, #2a2a35);
    border-radius: 8px;
}
.fc3d-editor-root .fc3d-framelbl {
    font: 600 10px ui-monospace, monospace; color: var(--c2c-subtext1, #bac2de);
    letter-spacing: .02em;
}
/* Range slider */
.fc3d-editor-root input[type=range].fc3d-slider {
    -webkit-appearance: none; appearance: none; height: 4px; border-radius: 999px;
    background: var(--c2c-surface1, #45475a); outline: none; cursor: pointer;
}
.fc3d-editor-root input[type=range].fc3d-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 13px; height: 13px; border-radius: 50%;
    background: var(--c2c-blue, #89b4fa); border: 2px solid var(--c2c-bg, #1a1a22);
    box-shadow: 0 1px 2px rgba(0,0,0,.4); transition: transform 100ms ease;
}
.fc3d-editor-root input[type=range].fc3d-slider:hover::-webkit-slider-thumb { transform: scale(1.18); }
.fc3d-editor-root input[type=range].fc3d-slider::-moz-range-thumb {
    width: 13px; height: 13px; border-radius: 50%; background: var(--c2c-blue, #89b4fa);
    border: 2px solid var(--c2c-bg, #1a1a22); cursor: pointer;
}
.fc3d-editor-root input[type=range].fc3d-slider::-moz-range-track {
    height: 4px; border-radius: 999px; background: var(--c2c-surface1, #45475a);
}
/* Canvas frame */
.fc3d-editor-root .fc3d-canvaswrap {
    border: 1px solid var(--c2c-surface0, #2a2a35); border-radius: 8px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.03), inset 0 6px 22px rgba(0,0,0,.28);
}
/* Sections + chip rows */
.fc3d-editor-root .fc3d-section-title {
    font: 600 9px ui-sans-serif, system-ui; letter-spacing: .08em; text-transform: uppercase;
    color: var(--c2c-sub, #7f849c); margin: 3px 0 1px;
}
.fc3d-editor-root .fc3d-chiprow { display: flex; flex-wrap: wrap; gap: 4px; }
/* Info / empty-state banner */
.fc3d-editor-root .fc3d-banner {
    display: flex; align-items: center; gap: 6px; padding: 5px 8px; border-radius: 6px;
    background: color-mix(in srgb, var(--c2c-blue, #89b4fa) 14%, transparent);
    color: var(--c2c-subtext1, #bac2de); font: 10px ui-sans-serif, system-ui;
    border: 1px solid color-mix(in srgb, var(--c2c-blue, #89b4fa) 28%, transparent);
}
/* Form controls (numeric inputs, selects) + disclosure summaries */
.fc3d-editor-root input[type=number], .fc3d-editor-root select {
    background: var(--c2c-bg2, #1a1a23); color: var(--c2c-fg, #cdd6f4);
    border: 1px solid var(--c2c-surface0, #2a2a35); border-radius: 5px; outline: none;
    transition: border-color var(--c2c-dur-fast,120ms) ease, box-shadow var(--c2c-dur-fast,120ms) ease;
}
.fc3d-editor-root input[type=number]:hover, .fc3d-editor-root select:hover { border-color: var(--c2c-surface2, #585b70); }
.fc3d-editor-root input[type=number]:focus, .fc3d-editor-root select:focus {
    border-color: var(--c2c-blue, #89b4fa);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--c2c-blue, #89b4fa) 30%, transparent);
}
.fc3d-editor-root details > summary { list-style: none; transition: color var(--c2c-dur-fast,120ms) ease; }
.fc3d-editor-root details > summary:hover { color: var(--c2c-fg, #cdd6f4); }
.fc3d-editor-root details > summary::-webkit-details-marker { display: none; }
/* Pill chips (expression presets) */
.fc3d-editor-root .fc3d-chip { border-radius: 999px !important; padding: 3px 11px !important; }
/* Themed scrollbars inside the editor */
.fc3d-editor-root *::-webkit-scrollbar { width: 8px; height: 8px; }
.fc3d-editor-root *::-webkit-scrollbar-thumb { background: var(--c2c-surface1, #45475a); border-radius: 999px; }
.fc3d-editor-root *::-webkit-scrollbar-track { background: transparent; }
@media (prefers-reduced-motion: reduce) {
    .fc3d-editor-root *, .fc3d-editor-root *::before, .fc3d-editor-root *::after {
        transition-duration: .001ms !important; animation-duration: .001ms !important;
    }
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
        domW.element.style.height = `${contentH()}px`;
        domW.element.style.minHeight = `${FC3D_MIN_H}px`;
        domW.element.style.maxHeight = `${FC3D_MAX_H}px`;
        domW.element.style.boxSizing = "border-box";
        domW.element.style.background = C.bg;
        domW.element.style.padding = "0";
    }
}

// Re-entry guard: setSize fires onResize → onResize calls _relayout →
// _relayout calls _fc3dSyncNodeSize → setSize again. Without this flag
// (and the 2px threshold below) the chain runs every frame and the page
// pegs at 11k rAF/sec. Touch-the-node crash root cause.
let _fc3dSyncInFlight = false;
function _fc3dSyncNodeSize(node) {
    if (!node || _fc3dSyncInFlight) return;
    _fc3dSyncInFlight = true;
    try {
        // Target = CURRENT width + its derived height. (computeSize() now has
        // LiteGraph min-clamp semantics — using it as a target would shrink
        // every node to the minimum width on each sync.)
        const cur = node.size || [0, 0];
        const w = Math.max(FC3D_NODE_W, cur[0] || FC3D_NODE_W);
        const editorH = node._faceOverlay?.getHeightForWidth?.(w)
            || node._faceOverlay?.getHeight?.() || FC3D_MIN_H;
        const sz = [w, _fc3dEditorTopPx(node) + Math.min(FC3D_MAX_H, Math.max(FC3D_MIN_H, editorH)) + 8];
        // Skip setSize when nothing meaningful changed — sub-pixel drift
        // from layout reflows is enough to round-trip onResize forever.
        if (Math.abs((sz?.[0] || 0) - (cur[0] || 0)) > 2 ||
            Math.abs((sz?.[1] || 0) - (cur[1] || 0)) > 2) {
            // Tag this setSize as OUR sync so onResize doesn't mistake it for
            // a user drag (that mistake was the snap-back/runaway source).
            node.__fc3dSyncing = true;
            try { node.setSize(sz); } finally { node.__fc3dSyncing = false; }
        }
        node.setDirtyCanvas?.(true, true);
    } catch (_) {}
    finally { _fc3dSyncInFlight = false; }
}

/** Measure the editor's top offset INSIDE the node, in graph units.
 *  The node body above the editor holds the title, I/O sockets and the
 *  visible detection widgets; converting a node-height drag into an editor
 *  height without subtracting this offset was why resize could never settle
 *  (the editor was told to be as tall as the whole node). */
function _fc3dEditorTopPx(node) {
    try {
        const domW = node.widgets?.find((w) => w.name === "face_overlay");
        // LiteGraph canvas mode: widgets carry their y inside the node.
        if (typeof domW?.last_y === "number" && domW.last_y > 8) return domW.last_y;
        // Vue/DOM mode: measure the element against the node's screen rect.
        const el = domW?.element;
        const c = app?.canvas;
        if (el && c?.ds && c.canvas) {
            const r = el.getBoundingClientRect();
            const rc = c.canvas.getBoundingClientRect();
            const graphY = (r.top - rc.top) / c.ds.scale - c.ds.offset[1];
            const top = graphY - node.pos[1];
            if (Number.isFinite(top) && top > 8 && top < 600) return top;
        }
    } catch (_) {}
    return 48;  // title + margins fallback
}

function makeDefaultMeta() {
    const zeros = Array.from({length:68}, () => [0, 0]);
    const selected = Array.from({length: 68}, (_, i) => i);
    const canon = CANONICAL.map((p) => [p[0], p[1]]);
    return {
        selected, eye_emph: [37, 38, 43, 44], d_norm: zeros,
        frames: [{ i: 0, ok: true, lms: canon }],
        strength: 1.0,
        pose: { format: "openpose_18", joint_names: POSE18_NAMES, edges: POSE18_EDGES_DEFAULT, frames: [_defaultPoseFrame()] },
        gaze: null, _synthetic: true,
    };
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
    b.classList.add("fc3d-btn");
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
    _fc3dEnsureSkinCss();

    // Middle-mouse / wheel-button pan over the DOM widget (ComfyUI graph is underneath).
    let _fc3dPanning = false;
    let _fc3dPanStart = null;
    const _fc3dPanApply = (e) => {
        const c = app?.canvas;
        if (!c?.ds || !_fc3dPanStart) return;
        c.ds.offset[0] = _fc3dPanStart.off[0] + (e.clientX - _fc3dPanStart.x);
        c.ds.offset[1] = _fc3dPanStart.off[1] + (e.clientY - _fc3dPanStart.y);
        c.setDirty?.(true, true);
        c.draw?.(true, true);
    };
    const _fc3dPanStop = () => {
        _fc3dPanning = false;
        _fc3dPanStart = null;
        window.removeEventListener("pointermove", _fc3dPanApply, true);
        window.removeEventListener("pointerup", _fc3dPanStop, true);
    };
    root.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;
        const c = app?.canvas;
        if (!c?.ds) return;
        _fc3dPanning = true;
        _fc3dPanStart = { x: e.clientX, y: e.clientY, off: c.ds.offset.slice() };
        e.preventDefault();
        e.stopPropagation();
        try { c.processMouseDown?.(e); } catch (_) {}
        window.addEventListener("pointermove", _fc3dPanApply, true);
        window.addEventListener("pointerup", _fc3dPanStop, true);
    }, true);
    root.addEventListener("pointerup", (e) => {
        if (e.button !== 1 || !_fc3dPanning) return;
        try { app.canvas?.processMouseUp?.(e); } catch (_) {}
        _fc3dPanStop();
    }, true);

    // ── Tab bar ──────────────────────────────────────────────────────
    const tabBar = _el("div",
        `display:flex;gap:2px;overflow:hidden;pointer-events:auto;flex:0 0 auto;`);
    tabBar.classList.add("fc3d-tabbar");
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
        const b = _el("button", "");
        b.classList.add("fc3d-tab");
        if (t.id === "face") b.classList.add("active");
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
        `position:relative;flex:0 0 auto;width:100%;overflow:hidden;background:#2a2a3d;` +
        `pointer-events:auto;cursor:crosshair;`);
    canvasWrap.classList.add("fc3d-canvaswrap");
    const cvs = _el("canvas", "display:block;width:100%;height:100%;cursor:crosshair;outline:none;");
    cvs.dataset.fc3dMainCanvas = "1";
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
        // Resizing clears the bitmap — schedule repaint (hook may call repaint too).
        requestAnimationFrame(() => { try { render(); drawTimeline(); } catch (_) {} });
        return true;
    }
    function _syncCanvasPx() {
        let side = Math.floor(canvasWrap.clientWidth || _canvasViewPx);
        // The square canvas must NEVER be taller than the height left after the
        // tabBar + transport + timeline + context strip, or growing a WIDE node
        // makes the square so tall it pushes the transport row and timeline out
        // of the overflow:hidden editor (the "resize hides the controls" bug).
        // Mirror the budget _fillAvailableHeight uses so every path agrees on
        // the canvas size — the single source of truth the resize paths lacked.
        try {
            const panel = ctxPanels[activeTab];
            let ctxH = 48;
            if (panel && panel.style.display !== "none") {
                const cap = FC3D_TAB_CTX_MAX[activeTab] ?? FC3D_CTX_SCROLL_MAX;
                ctxH = Math.min(cap, Math.max(48, panel.scrollHeight + 4));
            }
            const hBudget = _editorH - _CHROME_PX - ctxH;
            if (Number.isFinite(hBudget) && hBudget >= CANVAS_MIN_PX) side = Math.min(side, hBudget);
        } catch (_) { /* pre-init: fall back to width-only sizing */ }
        side = Math.max(CANVAS_MIN_PX, Math.min(CANVAS_MAX_PX, side));
        return _applyCanvasViewPx(side);
    }
    _applyCanvasViewPx(FC3D_CANVAS_VIEW_PX);
    cvs.tabIndex = 0;
    cvs.title = "drag to edit \u00b7 arrow keys step frame \u00b7 R reset frame";
    /** Undo LiteGraph zoom scale so drags land on the correct landmark. */
    function eventCanvas(e) {
        // Map a pointer event to INTERNAL canvas pixels. Everything is drawn in
        // [0..cvs.width]×[0..cvs.height]; the element is rendered at the
        // (possibly ComfyUI-zoomed) visual size r.width×r.height. So the
        // fraction across the visual rect × the internal resolution gives the
        // exact draw-space coordinate — correct at ANY zoom and regardless of
        // CSS-size-vs-internal-resolution mismatch.
        // (Bug: using cvs.offsetWidth mapped clicks into LAYOUT px, not draw px,
        //  so the pointer landed left of / off the joints when the two differ.)
        const r = cvs.getBoundingClientRect();
        const sx = cvs.width  / Math.max(1, r.width);
        const sy = cvs.height / Math.max(1, r.height);
        return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
    }
    cvs.addEventListener("focus", () => { cvs.style.boxShadow = `0 0 0 2px ${C.accent}`; });
    cvs.addEventListener("blur",  () => { cvs.style.boxShadow = "none"; });
    canvasWrap.appendChild(cvs);
    root.appendChild(canvasWrap);

    // ── Transport bar ───────────────────────────────────────────────
    const transport = _el("div",
        `display:flex;align-items:center;gap:4px;padding:4px 6px;flex:0 0 auto;pointer-events:auto;`);
    transport.classList.add("fc3d-transport");
    const btnPrev = _btn("\u25C0","Previous frame"); btnPrev.style.padding="2px 6px";
    const frameLbl = _el("span",`min-width:64px;text-align:center;`);
    frameLbl.classList.add("fc3d-framelbl");
    frameLbl.textContent = "f 0 / 0";
    const slider = _el("input",`flex:1;min-width:60px;`);
    slider.classList.add("fc3d-slider");
    slider.type="range"; slider.min="0"; slider.max="0"; slider.value="0";
    const btnNext = _btn("\u25B6","Next frame"); btnNext.style.padding="2px 6px";
    const btnUndo = _btn("\u21B6","Undo (Ctrl+Z)"); btnUndo.disabled=true;
    const btnRedo = _btn("\u21B7","Redo (Ctrl+Shift+Z)"); btnRedo.disabled=true;
    const btnReset = _btn("\u21BA","Reset this frame",C.err_bg);
    // Slice B: live face-warp preview toggle \u2014 deforms the input face image by
    // the rotated landmarks (head turns as you drag head pose), zero-GPU. The
    // queued ExpressionEditor remains the high-quality path.
    const btnWarp = _btn("\u25D1 warp","Live face-warp preview: the input face turns as you drag head pose (needs an input image + a queued frame)");
    btnWarp.style.padding="2px 6px";
    let _warpEnabled=false;
    btnWarp.addEventListener("click",()=>{_warpEnabled=!_warpEnabled;btnWarp.style.background=_warpEnabled?C.accent:"";try{render();}catch(_){}});
    transport.append(btnPrev,frameLbl,slider,btnNext,btnUndo,btnRedo,btnReset,btnWarp);
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

    // -- Face context: live render preview + collapsed numeric editor
    const ctxFace = _el("div", _ctxPanelStyle);
    // Live WYSIWYG thumbnail (Face-Director plan P5): every server resync can
    // return the EDITED frame rendered through the same overlay renderer as
    // the node's preview_image output. Drag a slider / DOF → see the result.
    const lpDetails = _el("details",
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:6px;overflow:hidden;margin-bottom:3px;`);
    lpDetails.open = false;  // closed by default — opt-in, so the Face tab stays compact
    const lpSummary = _el("summary",
        `padding:6px 10px;cursor:pointer;font:11px ui-sans-serif;color:${C.dim};user-select:none;`);
    lpSummary.textContent = "▸ Live render preview";
    lpDetails.addEventListener("toggle", () => {
        lpSummary.textContent = (lpDetails.open ? "▾ " : "▸ ") + "Live render preview";
        if (lpDetails.open) { try { _scheduleServerResync(); } catch (_) {} }
        _relayout();
    });
    const lpImg = _el("img",
        `display:block;width:100%;max-width:256px;margin:0 auto;border-radius:4px;` +
        `background:${C.bg};min-height:96px;object-fit:contain;`);
    lpImg.alt = "";
    const lpHint = _el("div",
        `padding:4px 8px;font:9px ui-sans-serif;color:${C.dim};text-align:center;`);
    lpHint.textContent = "queue once, then edits re-render here live";
    const lpBody = _el("div", `padding:6px;`);
    lpBody.append(lpImg, lpHint);
    lpDetails.append(lpSummary, lpBody);
    ctxFace.appendChild(lpDetails);
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
            `padding:3px 11px;font:10px ui-sans-serif;cursor:pointer;` +
            `background:${name==="Neutral"?C.tab_active:C.btn_off_bg};color:${C.text};border:1px solid ${C.border};`);
        b.classList.add("fc3d-btn", "fc3d-chip");
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
            try { _refreshFacs(); } catch (_) {}
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

    // ── P2 (Face-Director plan): per-AU dampening ───────────────────
    // A ceiling per AU (capped by the global expression_clamp). Default
    // 1.5 = "use global". Anything lower damps that single AU without
    // touching the others; written to expression_clamp_per_axis_json
    // (bare-number form), consumed by _apply_per_axis_clamp() backend.
    const FC3D_GLOBAL_CAP = 1.5;
    function _parsePerAxisClamp(){
        let raw = readParam(node, "expression_clamp_per_axis_json");
        if (!raw || !String(raw).trim()) return {};
        try { const o = JSON.parse(raw); return (o && typeof o === "object") ? o : {}; }
        catch (_) { return {}; }
    }
    function _capForAxis(map, id){
        const v = map[id];
        if (typeof v === "number") return v;
        if (v && typeof v === "object" && typeof v.clamp === "number") return v.clamp;
        if (v && typeof v === "object" && typeof v.max === "number") return v.max;
        return FC3D_GLOBAL_CAP;
    }
    function _writePerAxisCap(id, cap){
        const map = _parsePerAxisClamp();
        const existing = map[id];
        if (cap >= FC3D_GLOBAL_CAP - 1e-6) {
            // back to global → drop the entry (or its cap, preserving gain)
            if (existing && typeof existing === "object") { delete existing.clamp; delete existing.max;
                if (!("gain" in existing) && !("scale" in existing)) delete map[id]; }
            else delete map[id];
        } else if (existing && typeof existing === "object") {
            existing.clamp = +cap.toFixed(2); delete existing.max;
        } else {
            map[id] = +cap.toFixed(2);
        }
        writeParam(node, "expression_clamp_per_axis_json", Object.keys(map).length ? JSON.stringify(map) : "");
        try { _scheduleServerResync(); } catch (_) {}
    }
    const _limitEls = {};
    const exprLimits = _el("details",
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;margin-top:3px;`);
    const exprLimSum = _el("summary",
        `padding:4px 8px;cursor:pointer;font:10px ui-sans-serif;color:${C.dim};user-select:none;`);
    exprLimSum.textContent = "▸ Per-AU limits (dampening)";
    exprLimits.appendChild(exprLimSum);
    // Self-bounded: a 12-row grid would push the editor taller than the
    // node and get clipped by the wrapper's overflow:hidden. Cap it and
    // let it scroll internally (opt-in power panel — internal scroll OK).
    const limGrid = _el("div", `display:grid;grid-template-columns:1fr 1fr;gap:2px 6px;padding:4px 6px;` +
        `max-height:128px;overflow-y:auto;overflow-x:hidden;`);
    function _addLimitRow(axis, parent){
        const row = _el("div", `display:grid;grid-template-columns:52px 1fr 30px;align-items:center;gap:2px;height:18px;`);
        const lbl = _el("span", `font:9px ui-sans-serif;color:${C.dim};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`);
        lbl.textContent = axis.label.split(" ").slice(-1)[0] || axis.id; lbl.title = "Max for " + axis.label;
        const sl = _el("input", `width:100%;height:10px;accent-color:${C.accent};cursor:pointer;`);
        sl.type = "range"; sl.min = "0"; sl.max = "1.5"; sl.step = "0.05"; sl.value = String(FC3D_GLOBAL_CAP);
        const val = _el("span", `font:9px ui-monospace,monospace;color:${C.dim};text-align:right;`);
        const _paint = (v) => { const capped = v < FC3D_GLOBAL_CAP - 1e-6;
            val.textContent = capped ? v.toFixed(2) : "max"; val.style.color = capped ? C.accent : C.dim; };
        sl.addEventListener("input", () => { const v = parseFloat(sl.value); _paint(v); _writePerAxisCap(axis.id, v); });
        sl.addEventListener("dblclick", () => { sl.value = String(FC3D_GLOBAL_CAP); _paint(FC3D_GLOBAL_CAP); _writePerAxisCap(axis.id, FC3D_GLOBAL_CAP); });
        row.append(lbl, sl, val); parent.appendChild(row);
        _limitEls[axis.id] = { sl, val, paint: _paint };
    }
    for (const axis of FACS_AXES) _addLimitRow(axis, limGrid);
    exprLimits.appendChild(limGrid);
    const limFoot = _el("div", "display:flex;justify-content:flex-end;padding:0 6px 5px;");
    const btnLimReset = _btn("Reset limits", "Clear all per-AU dampening", C.input_bg);
    btnLimReset.style.cssText = "font-size:9px;padding:2px 6px;";
    btnLimReset.addEventListener("click", () => {
        writeParam(node, "expression_clamp_per_axis_json", "");
        for (const id in _limitEls) { _limitEls[id].sl.value = String(FC3D_GLOBAL_CAP); _limitEls[id].paint(FC3D_GLOBAL_CAP); }
        try { _scheduleServerResync(); } catch (_) {}
    });
    limFoot.appendChild(btnLimReset);
    exprLimits.appendChild(limFoot);
    function _refreshLimits(){ const map = _parsePerAxisClamp();
        for (const axis of FACS_AXES){ const el = _limitEls[axis.id]; if (!el) continue;
            const cap = _capForAxis(map, axis.id); el.sl.value = String(cap); el.paint(cap); } }

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
    exprLimits.open = false;
    exprLimits.addEventListener("toggle", () => {
        exprLimSum.textContent = (exprLimits.open ? "▾ " : "▸ ") + "Per-AU limits (dampening)";
        if (exprLimits.open) _refreshLimits();
        _relayout();
    });
    ctxExpr.appendChild(exprLimits);
    _refreshLimits();

    // ── P6 (Face-Director plan): FACS keyframe lanes ────────────────
    // An After-Effects-style 12-lane mini-timeline. Each AU gets a row;
    // every keyframe is a dot whose height/brightness encodes the coeff
    // (above baseline = positive, below = negative). The shared playhead
    // tracks the current frame; click any lane to scrub there.
    const facsDetails = _el("details",
        `background:${C.canvas_bg};border:1px solid ${C.border};border-radius:5px;margin-top:3px;`);
    const facsSum = _el("summary",
        `padding:4px 8px;cursor:pointer;font:10px ui-sans-serif;color:${C.dim};user-select:none;`);
    facsSum.textContent = "▸ FACS keyframe lanes";
    facsDetails.appendChild(facsSum);
    const facsWrap = _el("div", "padding:4px 6px;");
    const facsCvs = _el("canvas", `width:100%;height:${FACS_AXES.length*11+4}px;display:block;border-radius:3px;cursor:pointer;`);
    facsCvs.width = 460; facsCvs.height = FACS_AXES.length * 11 + 4;
    facsWrap.appendChild(facsCvs);
    const facsHint = _el("div", `font:8px ui-sans-serif;color:${C.dim};padding:2px 0 0;`);
    facsHint.textContent = "click a lane to scrub · dot height = coeff · ▲ above = +, ▼ below = −";
    facsWrap.appendChild(facsHint);
    facsDetails.appendChild(facsWrap);
    const _FACS_LBL_PX = 46;
    function drawFacsLanes() {
        const ctx = facsCvs.getContext("2d"), W = facsCvs.width, H = facsCvs.height;
        ctx.fillStyle = C.canvas_bg; ctx.fillRect(0, 0, W, H);
        const n = _frameCount();
        const trackX = _FACS_LBL_PX, trackW = Math.max(1, W - _FACS_LBL_PX - 2);
        const laneH = (H - 2) / FACS_AXES.length;
        const data = _parseCoeffsJson();
        const fx = (f) => (n <= 1 ? trackX + trackW / 2 : trackX + (f / (n - 1)) * trackW);
        FACS_AXES.forEach((a, li) => {
            const top = 2 + li * laneH, cy = top + laneH / 2;
            ctx.fillStyle = li % 2 ? "rgba(255,255,255,0.025)" : "rgba(0,0,0,0.12)";
            ctx.fillRect(0, top, W, laneH);
            ctx.fillStyle = C.dim; ctx.font = "8px ui-sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
            ctx.fillText((a.label.split(" ").slice(-1)[0] || a.id).slice(0, 8), 3, cy);
            ctx.strokeStyle = C.border; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(trackX, cy); ctx.lineTo(trackX + trackW, cy); ctx.stroke();
            for (const fk in data.frames) {
                const f = parseInt(fk, 10);
                if (!(f >= 0 && f < Math.max(1, n))) continue;
                const v = Number(data.frames[fk][a.id] || 0);
                if (Math.abs(v) < 0.001) continue;
                const mag = Math.min(1, Math.abs(v)), x = fx(f);
                ctx.fillStyle = v >= 0 ? C.sel : C.gaze_l;
                ctx.globalAlpha = 0.45 + mag * 0.55;
                ctx.beginPath(); ctx.arc(x, cy - v * (laneH * 0.34), 1.5 + mag * 2.2, 0, Math.PI * 2); ctx.fill();
                ctx.globalAlpha = 1;
            }
        });
        if (n > 0) {
            const xp = fx(state.frame);
            ctx.strokeStyle = C.accent; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xp, 1); ctx.lineTo(xp, H - 1); ctx.stroke();
        }
    }
    function _refreshFacs() { if (facsDetails.open) { try { drawFacsLanes(); } catch (_) {} } }
    facsCvs.addEventListener("click", (ev) => {
        const n = _frameCount(); if (n <= 0) return;
        const r = facsCvs.getBoundingClientRect();
        const mx = (ev.clientX - r.left) * (facsCvs.width / r.width);
        const t = Math.max(0, Math.min(1, (mx - _FACS_LBL_PX) / Math.max(1, facsCvs.width - _FACS_LBL_PX - 2)));
        _gotoFrame(Math.round(t * Math.max(0, n - 1)));
        drawFacsLanes();
    });
    facsDetails.open = false;
    facsDetails.addEventListener("toggle", () => {
        facsSum.textContent = (facsDetails.open ? "▾ " : "▸ ") + "FACS keyframe lanes";
        if (facsDetails.open) drawFacsLanes();
        _relayout();
    });
    ctxExpr.appendChild(facsDetails);

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
    // P4 (Face-Director plan): single "look-at target" mode. Toggle ON, then
    // drag one reticle on the face canvas — BOTH eyes converge on that point
    // (canvasToGaze run per-eye), the natural way humans fixate on a target.
    const lookAtBtn = _el("button",
        `font:9px ui-sans-serif;padding:4px 6px;border:1px solid ${C.border};border-radius:4px;cursor:pointer;` +
        `background:${C.btn_off_bg};color:${C.text};align-self:center;white-space:nowrap;`);
    lookAtBtn.textContent="🎯 Look-at";
    lookAtBtn.title="Look-at target: drag one point on the face — both eyes converge on it";
    lookAtBtn.addEventListener("click",()=>{
        gstate.lookAtMode=!gstate.lookAtMode;
        lookAtBtn.style.background=gstate.lookAtMode?C.accent:C.btn_off_bg;
        lookAtBtn.style.color=gstate.lookAtMode?C.bg:C.text;
        if(!gstate.lookAtMode)gstate.lookAtPos=null;
        render();
    });
    const gazeCenterCol = _el("div","display:flex;flex-direction:column;align-items:center;gap:4px;");
    gazeCenterCol.append(linkBtn,lookAtBtn);
    ctxGaze.append(gimbalL.el,gazeCenterCol,gimbalR.el);
    ctxPanels.gaze = ctxGaze;

    // Apply a look-at target: project the canvas point to per-eye gaze so
    // both eyes fixate on it, then write through the normal override path.
    function _applyLookAt(mx,my){
        const gl=canvasToGaze(mx,my,"l"),gr=canvasToGaze(mx,my,"r");
        _setGazeForFrame(state.frame,"l",gl.yaw,gl.pitch);
        _setGazeForFrame(state.frame,"r",gr.yaw,gr.pitch);
        gstate.lookAtPos=[mx,my];
        _drawGimbals();render();
        try{_scheduleServerResync();}catch(_){}
    }
    function _drawLookAt(ctx,W,H){
        if(!(activeTab==="gaze"&&gstate.lookAtMode))return;
        const lms=landmarksForFrame(state.frame);
        const[lxN,lyN]=_eyeCentroid(lms,"l"),[rxN,ryN]=_eyeCentroid(lms,"r");
        const[lx,ly]=denormToCanvas(lxN,lyN,W,H),[rx,ry]=denormToCanvas(rxN,ryN,W,H);
        let tx,ty;
        if(gstate.lookAtPos){[tx,ty]=gstate.lookAtPos;}
        else{tx=(lx+rx)/2;ty=(ly+ry)/2-H*0.05;}   // default: just above the eye line
        ctx.save();
        ctx.strokeStyle=C.accent;ctx.globalAlpha=0.55;ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
        ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(tx,ty);ctx.moveTo(rx,ry);ctx.lineTo(tx,ty);ctx.stroke();
        ctx.setLineDash([]);ctx.globalAlpha=1;
        ctx.strokeStyle=C.accent;ctx.lineWidth=2;
        ctx.beginPath();ctx.arc(tx,ty,9,0,Math.PI*2);ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(tx-13,ty);ctx.lineTo(tx-4,ty);ctx.moveTo(tx+4,ty);ctx.lineTo(tx+13,ty);
        ctx.moveTo(tx,ty-13);ctx.lineTo(tx,ty-4);ctx.moveTo(tx,ty+4);ctx.lineTo(tx,ty+13);ctx.stroke();
        ctx.fillStyle=C.accent;ctx.beginPath();ctx.arc(tx,ty,2.5,0,Math.PI*2);ctx.fill();
        ctx.restore();
    }

    // -- Pose context: 3D editor + collapsed advanced
    const ctxPose = _el("div", _ctxPanelStyle + "display:flex;flex-direction:column;gap:2px;");
    const btn3D = _btn("3D Editor\u2026","Open the 3D editor \u2014 pose head/face or the body skeleton in 3D (loads Three.js on demand)");
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

    // Dedicated full-width host for the 3D Head Editor — sits BELOW the
    // context strip, OUTSIDE the height-capped ctxScroll, so the Three.js
    // canvas gets real width+height and never collapses the node chrome.
    const panel3D = _el("div", "width:100%;flex:0 0 auto;overflow:hidden;");
    panel3D.style.display = "none";
    root.appendChild(panel3D);

    // ── Resize grip (bottom-right) ─────────────────────────────────
    // The editor root covers the node's bottom edge, so LiteGraph's own
    // corner hit-test is unreachable and the node could never be shrunk by
    // mouse. This grip drives the node resize directly (pointer capture →
    // node.size + editor refill), independent of LiteGraph hit-testing.
    root.style.position = "relative";
    const grip = _el("div",
        `position:absolute;right:0;bottom:0;width:20px;height:20px;cursor:nwse-resize;` +
        `z-index:5;border-bottom-right-radius:8px;opacity:.9;` +
        `background:linear-gradient(135deg, transparent 50%, ${C.accent} 50%);`);
    grip.title = "Drag to resize (scales the whole editor proportionally)";
    grip.addEventListener("mouseenter", () => { grip.style.opacity = "1"; });
    grip.addEventListener("mouseleave", () => { grip.style.opacity = ".65"; });
    root.appendChild(grip);
    let _gripStart = null;
    const _gripMove = (e) => {
        if (!_gripStart) return;
        const s = app?.canvas?.ds?.scale || 1;
        const topPx = _gripStart.top;
        // POINTS-EDITOR model: the drag drives ONE scale factor (the larger of
        // the two axis deltas, so diagonal/vertical/horizontal drags all feel
        // natural), that factor drives WIDTH, and width derives everything
        // else through _derivedLayout(). Height is never a free variable, so
        // the layout cannot ratchet or reflow randomly.
        const kx = (_gripStart.w + (e.clientX - _gripStart.x) / s) / Math.max(1, _gripStart.w);
        const ky = (_gripStart.h + (e.clientY - _gripStart.y) / s) / Math.max(1, _gripStart.h);
        const k = Math.max(0.3, Math.abs(kx - 1) >= Math.abs(ky - 1) ? kx : ky);
        const nw = Math.max(FC3D_NODE_W, Math.round(_gripStart.w * k));
        const editorH = _applyLayout(nw);
        _fc3dWriteNodeSize(node, nw, topPx + editorH + 8);
        node.setDirtyCanvas?.(true, true);
        e.preventDefault();
    };
    const _gripUp = () => {
        _gripStart = null;
        window.removeEventListener("pointermove", _gripMove, true);
        window.removeEventListener("pointerup", _gripUp, true);
    };
    grip.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        _gripStart = {
            x: e.clientX, y: e.clientY,
            w: node.size?.[0] || FC3D_NODE_W,
            h: node.size?.[1] || FC3D_MIN_H,
            top: _fc3dEditorTopPx(node),
        };
        window.addEventListener("pointermove", _gripMove, true);
        window.addEventListener("pointerup", _gripUp, true);
        e.preventDefault();
        e.stopPropagation();
    });

    // Resize observer: must be rAF-debounced and width-thresholded.
    // Without these guards a touch-the-node interaction creates a feedback
    // storm (callback writes canvasWrap.style.height → parent reflow →
    // canvasWrap.clientWidth shifts → callback fires again → ...). The
    // browser surfaces the storm as "ResizeObserver loop completed with
    // undelivered notifications" and the page paint thread hangs.
    let _resizeObs = null;
    let _roPending = false;
    let _roLastW = 0;
    try {
        _resizeObs = new ResizeObserver(() => {
            if (_roPending) return;
            _roPending = true;
            requestAnimationFrame(() => {
                _roPending = false;
                const w = canvasWrap.clientWidth || 0;
                // 2px hysteresis kills micro-jitter from sub-pixel reflow.
                if (Math.abs(w - _roLastW) < 2) return;
                _roLastW = w;
                if (_syncCanvasPx()) {
                    try { render(); drawTimeline(); } catch (_) {}
                }
            });
        });
        _resizeObs.observe(canvasWrap);
    } catch (_) {}

    // ── Dynamic node height (width-derived, points-editor model) ─────
    // 2026-06-12: the old multi-path height negotiation (user-height state
    // machine, height-as-input onResize, competing _fillAvailableHeight) was
    // replaced by ONE derivation: _derivedLayout(width). Height is no longer
    // an input ANYWHERE, which is what ended the ratchet/runaway/snap-back
    // family of bugs for good.
    let _editorH = 400;
    // Height the open 3D editor adds to the node (0 = closed). The 3D editor
    // mounts in its OWN full-width panel below the context strip — NOT inside
    // the height-capped ctxScroll (that was the "3D opens → everything
    // vanishes" bug). The derivation reserves this so the node grows for it.
    let _editor3dH = 0;
    const FC3D_EDITOR3D_PX = 360;
    let _lastLayoutH = 0;
    let _relayoutTimer = 0;
    // Measured chrome above/below the canvas: tabBar ~32 + gap 4 + transport 30 + gap 4 + timeline 24 + padding 10 = 104
    const _CHROME_PX = 32 + 4 + 30 + 4 + 24 + 10;

    /**
     * SINGLE SOURCE OF TRUTH for the editor layout (points-editor model).
     * Height is a pure function of WIDTH + active tab content:
     *     side    = clamp(nodeW − 52,  CANVAS_MIN..CANVAS_MAX)   (square face canvas)
     *     ctxH    = measured context-strip height (per-tab cap)
     *     editorH = clamp(side + chrome + ctxH, FC3D_MIN..MAX)
     * Width is the ONLY free variable. Every resize path (native edge drag,
     * grip drag, computeSize, tab switch, panel toggle) calls this same
     * derivation, so the layout can never ratchet, runaway, or reflow
     * "randomly" — exactly how points_bbox_editor.resizeForImage() works.
     */
    function _derivedLayout(nodeW) {
        const w = Math.max(FC3D_NODE_W, nodeW || node.size?.[0] || FC3D_NODE_W);
        const panel = ctxPanels[activeTab];
        let ctxH = 48;
        if (panel && panel.style.display !== "none") {
            const tabCap = FC3D_TAB_CTX_MAX[activeTab] ?? FC3D_CTX_SCROLL_MAX;
            const minH = activeTab === "set" ? 120 : 48;
            ctxH = Math.min(tabCap, Math.max(minH, panel.scrollHeight + 4));
        }
        const side = Math.max(CANVAS_MIN_PX, Math.min(CANVAS_MAX_PX, w - 52));
        // +14 bottom breathing room: without it the node's border sat ON the
        // context panel's last row (the half-covered "Clear" button report).
        // + _editor3dH: the full-width 3D editor panel (0 when closed).
        const editorH = Math.min(FC3D_MAX_H, Math.max(FC3D_MIN_H, side + _CHROME_PX + ctxH + 14 + _editor3dH));
        return { w, side, ctxH, editorH };
    }

    /** Apply the width-derived layout to the DOM. Returns the editor height. */
    function _applyLayout(nodeW) {
        const L = _derivedLayout(nodeW);
        _applyCanvasViewPx(L.side);
        if (ctxScroll) {
            ctxScroll.style.maxHeight = `${L.ctxH + 4}px`;
            ctxScroll.style.flex = "0 1 auto";
        }
        _editorH = L.editorH;
        root.style.height = `${L.editorH}px`;
        root.style.minHeight = `${FC3D_MIN_H}px`;
        root.style.maxHeight = `${FC3D_MAX_H}px`;
        return L.editorH;
    }

    // Back-compat shim: legacy callers pass (w, h) — height is now DERIVED
    // from width, so the h argument is ignored by design.
    function _fillAvailableHeight(wantW, _ignoredH) {
        return _applyLayout(wantW);
    }

    function _relayout() {
        _applyLayout(node.size?.[0] || FC3D_NODE_W);
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
            tabBtns[t.id].classList.toggle("active", t.id === id);
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
    const gstate = { dragEye:null, hoverEye:null, lookAtMode:false, lookAtPos:null, dragLookAt:false };
    // Seed the pose skeleton from the synthetic default so the Pose tab shows a
    // manipulable figure immediately (real data later replaces this via update()).
    const pstate = { edges:POSE18_EDGES_DEFAULT, names:POSE18_NAMES, frames:(state.meta?.pose?.frames||[]).map(fr=>fr?{...fr,kps:(fr.kps||[]).map(p=>Array.isArray(p)?[p[0],p[1]]:p)}:fr), dragJ:-1, hoverJ:-1 };
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
    function _refreshSliders(){const coeffs=_coeffsForFrame(state.frame);for(const a of FACS_AXES){const el=_sliderEls[a.id];if(!el)continue;const v=coeffs[a.id]||0;el.sl.value=String(v);el.val.textContent=v.toFixed(2);el.val.style.color=Math.abs(v)>0.01?C.sel:C.dim;}try{_refreshFacs();}catch(_){}}
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

    // ── Input-frame underlay ─────────────────────────────────────────
    // Legacy URL-list path (kept for setVideoFrames API) PLUS a lazy
    // server-fetched underlay: the node caches the INPUT frames and serves
    // them one-at-a-time from /c2c/fc3d_underlay so we draw input + pose
    // aligned, frame-by-frame, with zero GPU. Single image → constant.
    let _videoFrames=[];
    function _setVideoFrames(urls){_videoFrames=[];for(const url of(urls||[])){const img=new Image();img.src=url;_videoFrames.push(img);}}
    let _underlayCfg=null;            // {available, node_id, n_frames}
    const _underlayCache={};          // frame(str) -> HTMLImageElement | null
    function _setUnderlay(cfg){
        _underlayCfg=(cfg&&cfg.available&&cfg.node_id)?cfg:null;
        for(const k in _underlayCache)delete _underlayCache[k];
    }
    function _underlayImg(f){
        if(!_underlayCfg)return null;
        const key=String(f);
        if(key in _underlayCache){const c=_underlayCache[key];return (c&&c.complete&&c.naturalWidth>0)?c:null;}
        const img=new Image();
        _underlayCache[key]=img;
        img.onload=()=>{try{render();}catch(_){}};
        img.onerror=()=>{_underlayCache[key]=null;};
        img.src=`/c2c/fc3d_underlay?node_id=${encodeURIComponent(_underlayCfg.node_id)}&frame=${f}`;
        return null;
    }

    // ── Rendering functions ─────────────────────────────────────────
    let _tlRafPending = false;
    function render() {
        if (cvs.width < 16 || cvs.height < 16) _applyCanvasViewPx(FC3D_TAB_CANVAS_PX[activeTab] ?? FC3D_CANVAS_VIEW_PX);
        if(!_tlRafPending){_tlRafPending=true;requestAnimationFrame(()=>{_tlRafPending=false;drawTimeline();});}
        const ctx=cvs.getContext("2d"),W=cvs.width,H=cvs.height;
        if (!ctx || W < 8 || H < 8) return;
        ctx.fillStyle="#2a2a3d";ctx.fillRect(0,0,W,H);
        switch(activeTab) {
            case "face": case "gaze": case "set":
                _renderFaceGaze(ctx,W,H,activeTab==="gaze"); break;
            case "expr": _renderExpr(ctx,W,H); break;
            case "pose": _renderPose(ctx,W,H); break;
        }
    }

    function _drawGrid(ctx,W,H){
        ctx.strokeStyle="#6b7a9c";ctx.lineWidth=1;
        for(let i=1;i<4;i++){
            ctx.beginPath();ctx.moveTo((i/4)*W,0);ctx.lineTo((i/4)*W,H);ctx.stroke();
            ctx.beginPath();ctx.moveTo(0,(i/4)*H);ctx.lineTo(W,(i/4)*H);ctx.stroke();
        }
    }
    function _drawVideoUnderlay(ctx,W,H){
        let img=_underlayCfg?_underlayImg(state.frame):null;
        if(!img&&_videoFrames.length>0){const fi=Math.min(state.frame,_videoFrames.length-1),v=_videoFrames[fi];if(v&&v.complete&&v.naturalWidth>0)img=v;}
        if(img){ctx.save();ctx.globalAlpha=0.45;ctx.drawImage(img,0,0,W,H);ctx.globalAlpha=1;ctx.restore();}
    }
    // ── Slice B: live face-warp preview ──────────────────────────────
    // Delaunay (Bowyer–Watson) on the NEUTRAL landmarks → fixed triangle
    // topology (cached), then per-triangle affine texture-map of the input
    // face image from neutral→edited landmark positions. Pure Canvas2D.
    let _faceTris=null, _faceTrisN=0;
    function _delaunay(pts){
        const n=pts.length; if(n<3)return [];
        let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
        for(const p of pts){if(p[0]<minx)minx=p[0];if(p[1]<miny)miny=p[1];if(p[0]>maxx)maxx=p[0];if(p[1]>maxy)maxy=p[1];}
        const dx=maxx-minx||1,dy=maxy-miny||1,dm=Math.max(dx,dy),mx=(minx+maxx)/2,my=(miny+maxy)/2;
        const sp=[[mx-20*dm,my-dm],[mx,my+20*dm],[mx+20*dm,my-dm]];   // super-triangle
        const P=pts.concat(sp), si=n, tris=[[si,si+1,si+2]];
        const circ=(a,b,c,px,py)=>{const ax=P[a][0],ay=P[a][1],bx=P[b][0],by=P[b][1],cx=P[c][0],cy=P[c][1];
            const d=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by)); if(Math.abs(d)<1e-9)return false;
            const ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d;
            const uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d;
            const r2=(ax-ux)**2+(ay-uy)**2; return (px-ux)**2+(py-uy)**2<=r2+1e-6;};
        for(let i=0;i<n;i++){
            const px=P[i][0],py=P[i][1],bad=[];
            for(let t=0;t<tris.length;t++){const[a,b,c]=tris[t];if(circ(a,b,c,px,py))bad.push(t);}
            const edges=[];
            for(const t of bad){const[a,b,c]=tris[t];edges.push([a,b],[b,c],[c,a]);}
            for(let k=bad.length-1;k>=0;k--)tris.splice(bad[k],1);
            // keep boundary (non-shared) edges
            for(let e=0;e<edges.length;e++){let shared=false;for(let f=0;f<edges.length;f++){if(e!==f&&((edges[e][0]===edges[f][0]&&edges[e][1]===edges[f][1])||(edges[e][0]===edges[f][1]&&edges[e][1]===edges[f][0]))){shared=true;break;}}if(!shared)tris.push([edges[e][0],edges[e][1],i]);}
        }
        return tris.filter(t=>t[0]<n&&t[1]<n&&t[2]<n);  // drop super-triangle
    }
    function _warpAffine(ctx,s0,s1,s2,d0,d1,d2){
        const m00=s1[0]-s0[0],m01=s2[0]-s0[0],m10=s1[1]-s0[1],m11=s2[1]-s0[1];
        const det=m00*m11-m01*m10; if(Math.abs(det)<1e-6)return false;
        const n00=d1[0]-d0[0],n01=d2[0]-d0[0],n10=d1[1]-d0[1],n11=d2[1]-d0[1];
        const a=(n00*m11-n01*m10)/det, c=(-n00*m01+n01*m00)/det;
        const b=(n10*m11-n11*m10)/det, d=(-n10*m01+n11*m00)/det;
        const e=d0[0]-(a*s0[0]+c*s0[1]), f=d0[1]-(b*s0[0]+d*s0[1]);
        ctx.setTransform(a,b,c,d,e,f); return true;
    }
    function _drawWarpedFace(ctx,W,H){
        if(!_warpEnabled)return false;
        const img=_underlayCfg?_underlayImg(state.frame):(_videoFrames.length?_videoFrames[Math.min(state.frame,_videoFrames.length-1)]:null);
        if(!img||!img.complete||!img.naturalWidth)return false;
        const base=_localBaseline?._faceLmFor?.(state.frame)||_localBaseline?.faceLm?.[state.frame];
        let warp=null; try{warp=landmarksForFrame(state.frame);}catch(_){}
        if(!Array.isArray(base)||!Array.isArray(warp)||base.length<68||warp.length<68)return false;
        // (re)compute the fixed triangulation from the neutral landmarks (canvas px)
        const baseC=base.map(p=>denormToCanvas(p[0],p[1],W,H));
        if(!_faceTris||_faceTrisN!==base.length){_faceTris=_delaunay(baseC);_faceTrisN=base.length;}
        const iw=img.naturalWidth,ih=img.naturalHeight;
        const src=base.map(p=>[p[0]*iw,p[1]*ih]);            // input-image px
        const dst=warp.map(p=>denormToCanvas(p[0],p[1],W,H));// canvas px
        const baseT=ctx.getTransform?ctx.getTransform():null;
        ctx.save();
        for(const[ia,ib,ic]of _faceTris){
            const s0=src[ia],s1=src[ib],s2=src[ic],d0=dst[ia],d1=dst[ib],d2=dst[ic];
            if(!s0||!d0||!s1||!d1||!s2||!d2)continue;
            ctx.save();
            ctx.beginPath();ctx.moveTo(d0[0],d0[1]);ctx.lineTo(d1[0],d1[1]);ctx.lineTo(d2[0],d2[1]);ctx.closePath();ctx.clip();
            if(_warpAffine(ctx,s0,s1,s2,d0,d1,d2))ctx.drawImage(img,0,0);
            ctx.restore();
            if(baseT)ctx.setTransform(baseT);else ctx.setTransform(1,0,0,1,0,0);
        }
        ctx.restore();
        if(baseT)ctx.setTransform(baseT);else ctx.setTransform(1,0,0,1,0,0);
        return true;
    }
    function _drawFaceWireframe(ctx,W,H,lms,alpha){
        ctx.globalAlpha=alpha||1;
        ctx.strokeStyle = state.meta?._synthetic ? "#e8ecf8" : "#f5f7ff";
        ctx.lineWidth = state.meta?._synthetic ? 3 : 2.5;
        for(const seg of SEGMENTS){const[a,b,closed]=seg;ctx.beginPath();for(let i=a;i<b;i++){const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);if(i===a)ctx.moveTo(x,y);else ctx.lineTo(x,y);}if(closed)ctx.closePath();ctx.stroke();}
        ctx.globalAlpha=1;
    }

    function _renderFaceGaze(ctx,W,H,gazeEmphasis) {
        _drawVideoUnderlay(ctx,W,H);
        const warped=_drawWarpedFace(ctx,W,H);   // Slice B live face-warp (if toggled)
        _drawGrid(ctx,W,H);
        if (state.meta?._synthetic && !warped) {
            ctx.fillStyle = C.dim; ctx.font="10px ui-sans-serif"; ctx.textAlign = "center";
            ctx.fillText("canonical preview — queue with pose_data for detected face", W / 2, 14);
        }
        const lms=landmarksForFrame(state.frame),sel=selectedSet(),emp=emphSet();
        // Over a live warp, fade the wireframe so the deformed face reads clearly.
        _drawFaceWireframe(ctx,W,H,lms,gazeEmphasis?0.4:(warped?0.35:1));
        for(let i=0;i<lms.length;i++){const[x,y]=denormToCanvas(lms[i][0],lms[i][1],W,H);const isSel=sel.has(i),isEmph=emp.has(i);const r=(state.hoverLm===i||state.dragLm===i)?6:isSel?4:2.5;ctx.beginPath();ctx.fillStyle=gazeEmphasis?(C.other+"80"):(isEmph?C.emph:(isSel?C.sel:C.other));ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
        if(!gazeEmphasis&&state.hoverLm>=0){const[x,y]=denormToCanvas(lms[state.hoverLm][0],lms[state.hoverLm][1],W,H);ctx.fillStyle=C.text;ctx.font="10px ui-monospace,monospace";ctx.textAlign="left";ctx.fillText(String(state.hoverLm),x+6,y-4);}
        // Gaze handles
        const handles=computeGazeHandles(lms,W,H);
        for(const h of handles){const isDrag=gstate.dragEye===h.eye,isHov=gstate.hoverEye===h.eye,col=isDrag?C.gaze_drag:(h.eye==="l"?C.gaze_l:C.gaze_r);
            ctx.strokeStyle=col;ctx.lineWidth=gazeEmphasis?2.5:2;ctx.beginPath();ctx.arc(h.ax,h.ay,gazeEmphasis?5:4,0,Math.PI*2);ctx.stroke();
            ctx.beginPath();ctx.moveTo(h.ax,h.ay);ctx.lineTo(h.tx,h.ty);ctx.stroke();
            const r=isDrag?7:(isHov?6:(gazeEmphasis?5:4));ctx.fillStyle=col;ctx.beginPath();ctx.arc(h.tx,h.ty,r,0,Math.PI*2);ctx.fill();}
        if(!gazeEmphasis) _drawHeadGimbal(ctx,W,H);  // P3 head gimbal on Face tab only (not Gaze/Settings)
        _drawLookAt(ctx,W,H);                         // P4 look-at reticle (gaze tab only, self-guarded)
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
        _drawVideoUnderlay(ctx,W,H);_drawGrid(ctx,W,H);
        if(!pstate.frames.length){ctx.fillStyle=C.dim;ctx.font="11px ui-sans-serif";ctx.textAlign="center";ctx.fillText("no pose data \u2014 queue once",W/2,H/2);_drawHeadGimbal(ctx,W,H);return;}
        const f=state.frame,kps=_bodyKpsForFrame(f);
        if(!kps){ctx.fillStyle=C.dim;ctx.font="11px ui-sans-serif";ctx.textAlign="center";ctx.fillText("no body keypoints for frame "+f,W/2,H/2);_drawHeadGimbal(ctx,W,H);return;}
        const fr=pstate.frames[f],imgW=Math.max(1,Number(fr.w)||1),imgH=Math.max(1,Number(fr.h)||1);
        const scale=Math.min(W/imgW,H/imgH),drawW=imgW*scale,drawH=imgH*scale,ox=(W-drawW)/2,oy=(H-drawH)/2;
        ctx.strokeStyle=C.border;ctx.lineWidth=1;ctx.strokeRect(ox+0.5,oy+0.5,drawW-1,drawH-1);
        const _toCvs=(xn,yn)=>[ox+xn*drawW,oy+yn*drawH];
        ctx.strokeStyle=C.pose_bone;ctx.lineWidth=2.5;
        for(const[a,b]of pstate.edges){const ka=kps[a],kb=kps[b];if(!ka||!kb)continue;const[x1,y1]=_toCvs(ka[0],ka[1]),[x2,y2]=_toCvs(kb[0],kb[1]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}
        for(let i=0;i<kps.length;i++){const k=kps[i];if(!k)continue;const[x,y]=_toCvs(k[0],k[1]);const isDrag=pstate.dragJ===i,isHover=pstate.hoverJ===i,r=isDrag?7:(isHover?6:4.5);ctx.beginPath();ctx.fillStyle=isDrag?C.pose_selected:C.pose_joint;ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.lineWidth=1;ctx.strokeStyle=C.canvas_bg;ctx.stroke();}
        if(pstate.hoverJ>=0&&kps[pstate.hoverJ]){const[x,y]=_toCvs(kps[pstate.hoverJ][0],kps[pstate.hoverJ][1]);ctx.fillStyle=C.text;ctx.font="10px ui-monospace,monospace";ctx.textAlign="left";ctx.fillText(`${pstate.hoverJ}:${pstate.names[pstate.hoverJ]||"?"}`,x+7,y-5);}
        _drawHeadGimbal(ctx,W,H);
    }

    // ── P3 (Face-Director plan): 3-axis head-pose gimbal rings ───────
    // Drawn over the Pose tab: yaw = horizontal ring, pitch = vertical
    // ring, roll = outer circle. Dragging a knob writes head_yaw_deg /
    // head_pitch_deg / head_roll_deg directly (the same widgets the 3D
    // editor drives) and triggers the live mirror + server preview.
    const _GIM = { drag: null };
    function _gimGeom(W, H) {
        const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.36;
        const d2r = Math.PI / 180;
        const yaw   = (Number(readParam(node, "head_yaw_deg"))   || 0) * d2r;
        const pitch = (Number(readParam(node, "head_pitch_deg")) || 0) * d2r;
        const roll  = (Number(readParam(node, "head_roll_deg"))  || 0) * d2r;
        return {
            cx, cy, R,
            kyaw:   [cx + Math.sin(yaw) * R,          cy + Math.cos(yaw) * R * 0.22],
            kpitch: [cx + Math.cos(pitch) * R * 0.22, cy - Math.sin(pitch) * R],
            kroll:  [cx + Math.sin(roll) * (R + 12),  cy - Math.cos(roll) * (R + 12)],
        };
    }
    function _drawHeadGimbal(ctx, W, H) {
        if (activeTab !== "pose" && activeTab !== "face") return;
        const g = _gimGeom(W, H);
        ctx.save();
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.5;
        // roll: outer circle
        ctx.strokeStyle = C.emph; ctx.globalAlpha = 0.35;
        ctx.beginPath(); ctx.arc(g.cx, g.cy, g.R + 12, 0, Math.PI * 2); ctx.stroke();
        // yaw: horizontal ellipse
        ctx.strokeStyle = C.accent;
        ctx.beginPath(); ctx.ellipse(g.cx, g.cy, g.R, g.R * 0.22, 0, 0, Math.PI * 2); ctx.stroke();
        // pitch: vertical ellipse
        ctx.strokeStyle = C.sel;
        ctx.beginPath(); ctx.ellipse(g.cx, g.cy, g.R * 0.22, g.R, 0, 0, Math.PI * 2); ctx.stroke();
        // knobs
        ctx.globalAlpha = 1;
        const knob = (p, color) => {
            ctx.beginPath(); ctx.fillStyle = color;
            ctx.arc(p[0], p[1], _GIM.drag ? 7 : 6, 0, Math.PI * 2); ctx.fill();
            ctx.lineWidth = 1.5; ctx.strokeStyle = C.canvas_bg; ctx.stroke();
        };
        knob(g.kyaw, C.accent); knob(g.kpitch, C.sel); knob(g.kroll, C.emph);
        ctx.font = "9px ui-sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = C.dim;
        ctx.fillText("yaw", g.kyaw[0] + 9, g.kyaw[1] + 3);
        ctx.fillText("pitch", g.kpitch[0] + 9, g.kpitch[1] + 3);
        ctx.fillText("roll", g.kroll[0] + 9, g.kroll[1] + 3);
        ctx.restore();
    }
    function _pickGimbal(mx, my) {
        if (activeTab !== "pose" && activeTab !== "face") return null;
        const g = _gimGeom(cvs.width, cvs.height);
        const hit = (p) => (mx - p[0]) ** 2 + (my - p[1]) ** 2 <= 12 * 12;
        if (hit(g.kroll))  return "roll";
        if (hit(g.kyaw))   return "yaw";
        if (hit(g.kpitch)) return "pitch";
        return null;
    }
    function _applyGimbalDrag(mx, my) {
        const g = _gimGeom(cvs.width, cvs.height);
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const r2d = 180 / Math.PI;
        if (_GIM.drag === "yaw") {
            const v = clamp(Math.asin(clamp((mx - g.cx) / g.R, -1, 1)) * r2d, -80, 80);
            writeParam(node, "head_yaw_deg", +v.toFixed(1));
        } else if (_GIM.drag === "pitch") {
            const v = clamp(Math.asin(clamp((g.cy - my) / g.R, -1, 1)) * r2d, -60, 60);
            writeParam(node, "head_pitch_deg", +v.toFixed(1));
        } else if (_GIM.drag === "roll") {
            const v = clamp(Math.atan2(mx - g.cx, g.cy - my) * r2d, -90, 90);
            writeParam(node, "head_roll_deg", +v.toFixed(1));
        }
        node.setDirtyCanvas?.(true, true);
        try { _scheduleLocalMirror(); _scheduleServerResync(); } catch (_) {}
        render();
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
        ctx.fillStyle="#2a2a3d";ctx.fillRect(0,0,W,H);
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
            if(activeTab==="gaze"&&gstate.lookAtMode){_applyLookAt(mx,my);gstate.dragLookAt=true;ev.stopPropagation();ev.preventDefault();return;}
            const eye=pickGazeHandle(mx,my);
            if(eye){gstate.dragEye=eye;ev.stopPropagation();ev.preventDefault();render();return;}
            if(activeTab==="face"){
                state.dragLm=pickLandmark(mx,my);
                if(state.dragLm>=0){
                    if(propagateMode){const ref=landmarksForFrame(state.frame)[state.dragLm];state._dragRef=[ref[0],ref[1]];const[s,e]=_propRange();state._dragRange=[s,e];const ov=parseOverrides(node),entry=_ensureRangeEntry(ov,s,e),cur=entry.delta[String(state.dragLm)];state._dragBase=Array.isArray(cur)&&cur.length===2?[Number(cur[0])||0,Number(cur[1])||0]:[0,0];}
                    ev.stopPropagation();ev.preventDefault();}
                else{const gax=_pickGimbal(mx,my);if(gax){_GIM.drag=gax;ev.stopPropagation();ev.preventDefault();render();return;}}
            }
        } else if(activeTab==="pose"){
            const gax=_pickGimbal(mx,my);
            if(gax){_GIM.drag=gax;ev.stopPropagation();ev.preventDefault();render();return;}
            pstate.dragJ=_pickJoint(mx,my);
            if(pstate.dragJ>=0){
                if(propagateMode){const ref=_bodyKpsForFrame(state.frame)[pstate.dragJ]||[0,0];pstate._dragRef=[ref[0],ref[1]];const[s,e]=_propRange();pstate._dragRange=[s,e];const ov=parsePoseOverrides(node),entry=_ensureRangeEntry(ov,s,e),cur=entry.delta[String(pstate.dragJ)];pstate._dragBase=Array.isArray(cur)&&cur.length===2?[Number(cur[0])||0,Number(cur[1])||0]:[0,0];}
                ev.stopPropagation();ev.preventDefault();render();}
        }
    });

    cvs.addEventListener("mousemove",ev=>{
        const[mx,my]=clientToCanvas(ev);
        if(gstate.dragLookAt){_applyLookAt(mx,my);ev.stopPropagation();return;}
        if(_GIM.drag){_applyGimbalDrag(mx,my);ev.stopPropagation();return;}
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

    const _endDrag=()=>{let dirty=false;if(state.dragLm>=0){state.dragLm=-1;dirty=true;}if(gstate.dragEye){gstate.dragEye=null;dirty=true;}if(gstate.dragLookAt){gstate.dragLookAt=false;dirty=true;}if(pstate.dragJ>=0){pstate.dragJ=-1;dirty=true;}if(_GIM.drag){_GIM.drag=null;dirty=true;}if(dirty)render();};
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
        if(_fc3dEditor){try{_fc3dEditor.destroy();}catch(_){}_fc3dEditor=null;if(_fc3dHost){try{_fc3dHost.remove();}catch(_){}_fc3dHost=null;}panel3D.style.display="none";_editor3dH=0;_relayout();btn3D.style.background=C.border;try{_persistSave?.();}catch(_){}return;}
        btn3D.style.background=C.accent;
        // Mount into the dedicated full-width panel (NOT the capped ctxScroll),
        // reserve its height in the layout, and relayout so the node grows.
        _fc3dHost=_el("div",`width:100%;height:${FC3D_EDITOR3D_PX}px;`);
        panel3D.innerHTML="";panel3D.style.display="block";panel3D.appendChild(_fc3dHost);
        _editor3dH=FC3D_EDITOR3D_PX+8;_relayout();
        const _wRead=(name)=>Number(readParam(node,name))||0;
        const _wWrite=(name,v)=>{let c=Number(v);if(!Number.isFinite(c))return;const opts=FC3D_PARAM_OPTS[name]||{};if(opts.min!==undefined)c=Math.max(opts.min,c);if(opts.max!==undefined)c=Math.min(opts.max,c);writeParam(node,name,c);node.setDirtyCanvas?.(true,true);};
        try{
            const mod=await import("./face_3d_editor.js");if(!_fc3dHost)return;
            _fc3dEditor=await mod.mount3DEditor(_fc3dHost,{
                theme:C,
                getLandmarks:()=>{try{return landmarksForFrame(state.frame);}catch(_){return null;}},
                getHeadPose:()=>({yaw:_wRead("head_yaw_deg"),pitch:_wRead("head_pitch_deg"),roll:_wRead("head_roll_deg"),tx:_wRead("head_tx"),ty:_wRead("head_ty"),tz:_wRead("head_tz"),scale:_wRead("head_scale"),jaw:_wRead("jaw_rot_deg"),neck_yaw:_wRead("neck_yaw_deg"),neck_pitch:_wRead("neck_pitch_deg")}),
                setHeadPose:partial=>{if(partial.yaw!==undefined)_wWrite("head_yaw_deg",partial.yaw);if(partial.pitch!==undefined)_wWrite("head_pitch_deg",partial.pitch);if(partial.roll!==undefined)_wWrite("head_roll_deg",partial.roll);if(partial.tx!==undefined)_wWrite("head_tx",partial.tx);if(partial.ty!==undefined)_wWrite("head_ty",partial.ty);if(partial.tz!==undefined)_wWrite("head_tz",partial.tz);if(partial.scale!==undefined)_wWrite("head_scale",partial.scale);if(partial.jaw!==undefined)_wWrite("jaw_rot_deg",partial.jaw);if(partial.neck_yaw!==undefined)_wWrite("neck_yaw_deg",partial.neck_yaw);if(partial.neck_pitch!==undefined)_wWrite("neck_pitch_deg",partial.neck_pitch);try{_scheduleLocalMirror();_scheduleServerResync();}catch(_){}},
                onClose:()=>{_fc3dEditor=null;if(_fc3dHost){try{_fc3dHost.remove();}catch(_){}_fc3dHost=null;}panel3D.style.display="none";_editor3dH=0;_relayout();btn3D.style.background=C.border;try{_persistSave?.();}catch(_){}},
                // ── Body skeleton (OpenPose-18) — 3D pose → 2D writeback ──
                bodyEdges:pstate.edges&&pstate.edges.length?pstate.edges:POSE18_EDGES_DEFAULT,
                bodyNames:pstate.names&&pstate.names.length?pstate.names:POSE18_NAMES,
                getLayer:()=>{try{return localStorage.getItem("mec.fc3d.layer")||"face";}catch(_){return "face";}},
                onLayerChange:(layer)=>{try{localStorage.setItem("mec.fc3d.layer",layer);}catch(_){}if((layer==="body"||layer==="camera")&&activeTab!=="pose")switchTab("pose");},
                getBodyJoints:()=>{try{return _bodyKpsForFrame(state.frame);}catch(_){return null;}},
                setBodyJoints:(map)=>{try{const ov=parsePoseOverrides(node),key=String(state.frame);if(!ov.frames)ov.frames={};if(!ov.frames[key])ov.frames[key]={};for(const k in map){const v=map[k];if(Array.isArray(v)&&v.length===2&&Number.isFinite(v[0])&&Number.isFinite(v[1]))ov.frames[key][String(k)]=[+Number(v[0]).toFixed(5),+Number(v[1]).toFixed(5)];}writePoseOverrides(node,ov);try{_scheduleLocalMirror();_scheduleServerResync();}catch(_){}render();}catch(_){}},
                clearBodyJoint:(idx)=>{try{const ov=parsePoseOverrides(node),key=String(state.frame);if(ov.frames?.[key]){delete ov.frames[key][String(idx)];if(!Object.keys(ov.frames[key]).length)delete ov.frames[key];writePoseOverrides(node,ov);try{_scheduleLocalMirror();_scheduleServerResync();}catch(_){}render();}}catch(_){}},
                height:FC3D_EDITOR3D_PX,
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
        // The Pose-tab head gimbal writes head_yaw/pitch; the skeleton's
        // head-cluster kps (nose/eyes/ears) should follow it too — so fold the
        // head yaw/pitch deltas into the neck rotation that drives those kps.
        // (Without this, dragging pitch/yaw/roll moved nothing on the pose.)
        const rScale=(now.head_scale||1)/(base.head_scale||1),
              dNYaw=((now.neck_yaw_deg-base.neck_yaw_deg)+(now.head_yaw_deg-base.head_yaw_deg))*Math.PI/180,
              dNPit=((now.neck_pitch_deg-base.neck_pitch_deg)+(now.head_pitch_deg-base.head_pitch_deg))*Math.PI/180;
        const cosR=Math.cos(dRoll),sinR=Math.sin(dRoll),cosY=Math.cos(dYaw),cosP=Math.cos(dPitch);
        const tzClamp=Math.max(-0.75,Math.min(3,dTz)),zoomTz=1/(1+tzClamp);
        const baseLm=_localBaseline.faceLm[frameIdx],faceFr=state.meta?.frames?.[frameIdx];
        if(baseLm&&faceFr){let cx=0,cy=0;for(const p of baseLm){cx+=p[0];cy+=p[1];}cx/=baseLm.length;cy/=baseLm.length;const s=rScale*zoomTz;let mnX=1,mxX=0;for(const p of baseLm){if(p[0]<mnX)mnX=p[0];if(p[0]>mxX)mxX=p[0];}const bbW=Math.max(1e-6,mxX-mnX);const dNorm=new Array(68),newLm=new Array(68);for(let i=0;i<baseLm.length;i++){let x=baseLm[i][0]-cx,y=baseLm[i][1]-cy;x*=s;y*=s;const rx=x*cosR-y*sinR,ry=x*sinR+y*cosR;let nx=rx*cosY+cx+dTx*bbW,ny=ry*cosP+cy+dTy*bbW;newLm[i]=[nx,ny];dNorm[i]=[(nx-baseLm[i][0])/bbW,(ny-baseLm[i][1])/bbW];}faceFr.lms=newLm;faceFr.d_norm=dNorm;}
        const basePose=_localBaseline.poseFrames?.[frameIdx],livePose=pstate.frames?.[frameIdx];
        if(basePose&&livePose&&Array.isArray(basePose.kps)&&Array.isArray(livePose.kps)&&livePose.kps.length===18){const neck=basePose.kps[1],baseKps=basePose.kps;if(Array.isArray(neck)&&neck.length>=2&&(Math.abs(dNYaw)>1e-3||Math.abs(dNPit)>1e-3||Math.abs(dRoll)>1e-3)){const nx=neck[0],ny=neck[1],headCluster=[0,14,15,16,17],cyN=Math.cos(dNYaw),cpN=Math.cos(dNPit),spN=Math.sin(dNPit);const newKps=baseKps.map(p=>Array.isArray(p)?p.slice():p);for(const idx of headCluster){const p=baseKps[idx];if(!Array.isArray(p))continue;const ox=(p[0]-nx)*cyN,oy=(p[1]-ny)*cpN-spN*Math.abs(p[1]-ny);const rx=ox*cosR-oy*sinR,ry=ox*sinR+oy*cosR;newKps[idx][0]=nx+rx;newKps[idx][1]=ny+ry;}livePose.kps=newKps;}else livePose.kps=baseKps.map(p=>Array.isArray(p)?p.slice():p);}
    }
    function _scheduleLocalMirror(){if(_mirrorRaf)return;_mirrorRaf=requestAnimationFrame(()=>{_mirrorRaf=0;try{_mirrorTransform(state.frame);}catch(_){}try{render();}catch(_){}});}
    async function _serverResync(){const seq=++_resyncSeq;if(_resyncInFlight)return;_resyncInFlight=true;try{const body={node_id:String(node.id),frame_idx:state.frame,return_image:!!lpDetails?.open,preview_size:256,..._readAllDOF()};const resp=await fetch("/c2c/fc3d_preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(seq!==_resyncSeq)return;if(resp.status===412){frameHint.textContent="queue node once for server preview";return;}if(!resp.ok)return;const data=await resp.json();if(seq!==_resyncSeq)return;if(data.image_b64&&lpImg){lpImg.src="data:image/png;base64,"+data.image_b64;lpHint.style.display="none";}const fIdx=data.frame_idx,faceArr=data.face_norm,bodyArr=data.body_kps;if(Array.isArray(faceArr)&&_localBaseline?.faceLm?.[fIdx]){const baseLm=_localBaseline.faceLm[fIdx];let mnX=1,mxX=0,mnY=1,mxY=0;for(const p of baseLm){if(p[0]<mnX)mnX=p[0];if(p[0]>mxX)mxX=p[0];if(p[1]<mnY)mnY=p[1];if(p[1]>mxY)mxY=p[1];}const bbW=Math.max(1e-6,mxX-mnX),bbH=Math.max(1e-6,mxY-mnY);const absLm=faceArr.map(p=>[mnX+p[0]*bbW,mnY+p[1]*bbH]);const faceFr=state.meta?.frames?.[fIdx];if(faceFr){faceFr.lms=absLm;faceFr.d_norm=absLm.map((p,i)=>[(p[0]-baseLm[i][0])/bbW,(p[1]-baseLm[i][1])/bbH]);}}if(Array.isArray(bodyArr)&&pstate.frames?.[fIdx]){pstate.frames[fIdx].kps=bodyArr.map(p=>Array.isArray(p)?p.slice():[NaN,NaN]);}try{render();}catch(_){}}catch(_){}finally{_resyncInFlight=false;}}
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
            try{_setUnderlay(meta&&typeof meta==="object"?meta.underlay:null);}catch(_){}
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
        // Width-derived (points-editor model): the editor's height is a pure
        // function of the node's CURRENT width + active tab content.
        getHeight: () => _derivedLayout(node.size?.[0]).editorH,
        /** Same derivation for an EXPLICIT width (computeSize min-clamp etc.). */
        getHeightForWidth: (w) => _derivedLayout(w).editorH,
        /** DEPRECATED — height is derived from width now; kept as a no-op so
         *  old callers (and stale deploys mid-sync) can't corrupt the layout. */
        setUserEditorHeight(_h) { /* height is f(width) — nothing to record */ },
        repaint() { try { render(); drawTimeline(); } catch (_) {} },
        onNodeResize(nodeW, _nodeH) {
            const applied = _applyLayout(nodeW);
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
    try{_captureBaseline();}catch(_){}   // baseline for the synthetic default so pitch/yaw/roll move coords pre-queue
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

    // Let pointer events fall through the ComfyUI .dom-widget WRAPPER in the
    // margins around the editor root. When the editor grows to the node's
    // bottom edge, the wrapper otherwise covers the LiteGraph resize corner
    // and swallows the mouse-down — making the node impossible to shrink.
    const _fixWrapperPointer = () => {
        try {
            const el = domW.element;
            const wrap = el?.closest?.(".dom-widget");
            if (wrap && wrap !== el) {
                wrap.style.pointerEvents = "none";
                el.style.pointerEvents = "auto";
                // Clip to the widget slot: when the editor's computed height
                // momentarily exceeds the node's layout slot (zoom changes,
                // tab switches, restores), the wrapper otherwise lets the DOM
                // spill OUTSIDE the node bounds over neighbouring nodes.
                wrap.style.overflow = "hidden";
            }
        } catch (_) {}
    };
    requestAnimationFrame(_fixWrapperPointer);
    setTimeout(_fixWrapperPointer, 400);

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

        // Re-entry guard: _fc3dSyncNodeSize (called from _relayout) calls
        // setSize → onResize → relayout → setSize. Without this guard the
        // chain runs every frame.
        let _inOnResize = false;
        const _origOnResize = node.onResize;
        node.onResize = function (size) {
            if (_inOnResize) return;
            _inOnResize = true;
            try {
                _origOnResize?.apply(this, arguments);
                // POINTS-EDITOR model: whatever edge/corner the user drags,
                // only the WIDTH is taken; the node height is derived from it
                // and written back. A pure-vertical drag just snaps back to
                // the derived height — height is never accepted as an input,
                // so nothing can ratchet "upper and upper" or reflow randomly.
                const sz = size || node.size;
                const nw = Math.max(FC3D_NODE_W, sz?.[0] || FC3D_NODE_W);
                const editorH = overlay.onNodeResize?.(nw) ?? FC3D_MIN_H;
                const topPx = _fc3dEditorTopPx(node);
                const cv = app?.canvas;
                const nativeDragLive = !node.__fc3dSyncing &&
                    (cv?.resizing_node === node || cv?.resizingNode === node);
                if (nativeDragLive) {
                    // Mid-drag: writing node.size now fights LiteGraph's drag
                    // anchor (the pointer-to-corner offset drifts and the drag
                    // under-applies). Let LiteGraph own the size until the
                    // pointer lifts, then snap once to the derived height.
                    if (!node.__fc3dDragEndArmed) {
                        node.__fc3dDragEndArmed = true;
                        const finalize = () => {
                            node.__fc3dDragEndArmed = false;
                            const w = Math.max(FC3D_NODE_W, node.size?.[0] || FC3D_NODE_W);
                            const eh = overlay.onNodeResize?.(w) ?? FC3D_MIN_H;
                            _fc3dWriteNodeSize(node, w, _fc3dEditorTopPx(node) + eh + 8);
                            node.setDirtyCanvas?.(true, true);
                        };
                        window.addEventListener("pointerup", finalize, { once: true, capture: true });
                    }
                } else {
                    _fc3dWriteNodeSize(node, nw, topPx + editorH + 8);
                }
            } finally { _inOnResize = false; }
        };
    }

    _fc3dInstances.add(node);
    _fc3dHookCanvasDraw();
    requestAnimationFrame(() => { try { overlay.repaint?.(); } catch (_) {} });
    setTimeout(() => { try { overlay.repaint?.(); } catch (_) {} }, 120);
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

        nodeType.prototype.computeSize = function (outW) {
            // LiteGraph treats this as the node's MINIMUM size during a corner
            // resize. The old `|| this.size[0]` fallback pinned the minimum at
            // the CURRENT width, so the node could never be dragged smaller.
            // Minimum is the true floor (FC3D_NODE_W) and its width-derived
            // height; when a width is supplied, answer for that width.
            const w = Math.max(FC3D_NODE_W, outW || FC3D_NODE_W);
            const editorH = this._faceOverlay?.getHeightForWidth?.(w)
                || this._faceOverlay?.getHeight?.()
                || FC3D_MIN_H;
            return [w, Math.min(FC3D_MAX_H, Math.max(FC3D_MIN_H, editorH))];
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
let _fc3dCanvasDrawHooked = false;
/** Keep the face canvas fresh across zoom changes WITHOUT repainting every
 *  frame. ComfyUI's fork runs a continuous draw loop, so repainting on every
 *  `app.canvas.draw` re-rendered 68 landmarks + the timeline for every node
 *  ~60x/second forever — pegging the main thread and freezing the UI
 *  (the "non-responsive" bug). The face bitmap only goes stale when the graph
 *  ZOOM (scale) changes; pan just moves the DOM widget and leaves the bitmap
 *  intact. So we repaint only when the scale (or the set of instances) changes
 *  — zero per-frame cost at idle. All data-driven repaints already happen via
 *  the overlay's own render()/drawTimeline() triggers. */
function _fc3dHookCanvasDraw() {
    if (_fc3dCanvasDrawHooked || !app?.canvas?.draw) return;
    _fc3dCanvasDrawHooked = true;
    const origDraw = app.canvas.draw.bind(app.canvas);
    let _lastScale = -1, _lastCount = -1;
    app.canvas.draw = function (force, skip_events) {
        const ret = origDraw(force, skip_events);
        const count = _fc3dInstances.size;
        if (count) {
            const scale = app.canvas?.ds?.scale ?? 1;
            if (scale !== _lastScale || count !== _lastCount) {
                _lastScale = scale; _lastCount = count;
                for (const n of _fc3dInstances) {
                    try { n._faceOverlay?.repaint?.(); } catch (_) {}
                }
            }
        }
        return ret;
    };
}
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
