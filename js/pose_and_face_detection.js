// PoseAndFaceDetectionV2: HIDE sub-options that don't apply to the current
// toggle/mode selection (user rule 2026-06-24: "show sub-options only when true,
// hide if false"). A sub-option appears only when its controlling boolean is ON
// (or the relevant crop_mode is selected); otherwise it is removed and the node
// shrinks. Re-applied whenever a controlling widget changes.
import { app } from "../../scripts/app.js";

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (!("__mec_origType" in w)) w.__mec_origType = w.type;
        if (!("__mec_origComputeSize" in w)) w.__mec_origComputeSize = w.computeSize;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.hidden = true;
        const el = w.element;
        if (el) {
            if (!("__mec_origElDisplay" in w)) w.__mec_origElDisplay = el.style.display;
            el.style.display = "none";
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if (!("__mec_origWrapDisplay" in w)) w.__mec_origWrapDisplay = wrap.style.display;
                wrap.style.display = "none";
            }
        }
    } else {
        if ("__mec_origType" in w) { w.type = w.__mec_origType; delete w.__mec_origType; }
        if ("__mec_origComputeSize" in w) {
            const cs = w.__mec_origComputeSize;
            if (cs === undefined) delete w.computeSize; else w.computeSize = cs;
            delete w.__mec_origComputeSize;
        }
        w.hidden = false;
        const el = w.element;
        if (el) {
            el.style.display = ("__mec_origElDisplay" in w) ? (w.__mec_origElDisplay ?? "") : "";
            delete w.__mec_origElDisplay;
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                wrap.style.display = ("__mec_origWrapDisplay" in w) ? (w.__mec_origWrapDisplay ?? "") : "";
                delete w.__mec_origWrapDisplay;
            }
        }
    }
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const useBlur    = !!get("use_blur_for_pose")?.value;
    const faceSmooth = !!get("use_face_smoothing")?.value;
    const constBox   = !!get("use_constant_face_box")?.value;
    const irisSmooth = !!get("use_iris_smoothing")?.value;
    const irisMethod = String(get("iris_smoothing_method")?.value ?? "one_euro");
    const gazeLock   = !!get("gaze_lock_eyes")?.value;
    const blendGaze  = !!get("use_blendshape_gaze")?.value;
    const cropMode   = String(get("crop_mode")?.value ?? "default");
    const jitterless = cropMode === "jitterless";
    const refSmooth  = cropMode === "reference_smooth";
    // expression_lock shares reference_smooth's branch but takes the centre
    // RAW, so the centre filter knobs are inert there — only the SIZE is
    // filtered, and crop_safety_margin is forced to 1.0 (no filter lag to
    // absorb, and inflating the box just costs face resolution).
    const exprLock   = cropMode === "expression_lock";
    const auto       = cropMode === "auto";
    const smoothing  = String(get("smoothing_method")?.value ?? "one_euro");
    const autoSm     = String(get("auto_smoothing_method")?.value ?? "legacy_ema");
    const cropActive = jitterless || auto;          // modes that BUILD their own box
    // Which shared temporal filter is actually running, per mode. auto has its
    // own selector and ignores smoothing_method entirely; reference_smooth and
    // jitterless share smoothing_method. Getting this wrong is what hid the
    // one_euro knobs from the modes that use them.
    const filtered   = jitterless || refSmooth || auto || exprLock;
    const method     = auto ? autoSm : smoothing;
    const oneEuro    = filtered && method === "one_euro";
    const gaussian   = filtered && method === "gaussian";
    const emaLike    = method === "ema" || method === "legacy_ema";
    const useClahe   = !!get("use_clahe")?.value;
    const gazeEngine = String(get("gaze_engine")?.value ?? "l2cs_gaze360");
    const eyeAlign   = String(get("eye_align_mode")?.value ?? "default");
    const auAmp      = Number(get("au_amplify")?.value ?? 1.0);
    const containOn  = !!get("crop_containment_check")?.value;
    const eyesOpen   = Number(get("force_eyes_open")?.value ?? 0.0) > 0.0;
    // Every engine EXCEPT the purely geometric one runs the Kalman stage.
    const kalmanEngine = ["blendshape_head_corrected", "blendshape_only",
                          "l2cs_gaze360", "l2cs_mpiigaze",
                          "pose_normalized_resnet50", "ethxgaze"].includes(gazeEngine);

    // visible[name] = true → show; everything not listed is always shown.
    const visible = {
        blur_radius:                useBlur,
        blur_sigma:                 useBlur,
        // CLAHE sub-options — the backend skips both when use_clahe is off
        clahe_clip_limit:           useClahe,
        clahe_grid_size:            useClahe,
        // face smoothing / constant-box only matter for crop_mode = auto
        use_face_smoothing:         auto,
        // strength is the EMA coefficient — it does nothing under one_euro or
        // gaussian, but it IS live for every ema-family method in every mode.
        face_smoothing_strength:    (auto && faceSmooth && emaLike) || (filtered && emaLike),
        use_constant_face_box:      auto,
        // Also the jitterless anchor size: with frame0_size unset, this widget
        // IS the locked crop size, so hiding it there left that mode's single
        // most important control unreachable.
        face_box_size_px:           (auto && constBox) || jitterless,
        // iris smoothing sub-options
        iris_smoothing_strength:    irisSmooth,
        iris_smoothing_method:      irisSmooth,
        iris_one_euro_min_cutoff:   irisSmooth && irisMethod === "one_euro",
        iris_one_euro_beta:         irisSmooth && irisMethod === "one_euro",
        // gaze sub-options
        gaze_lock_strength:         gazeLock,
        gaze_one_euro_min_cutoff:   blendGaze,
        gaze_one_euro_beta:         blendGaze,
        gaze_max_yaw_deg:           blendGaze,
        gaze_max_pitch_deg:         blendGaze,
        // manual-keyframe crop controls only matter for crop_mode = jitterless
        frame0_cx:                  jitterless,
        frame0_cy:                  jitterless,
        frame0_size:                jitterless,
        keyframes_json:             jitterless,
        // auto picks its filter with auto_smoothing_method instead
        smoothing_method:           jitterless || refSmooth || exprLock,
        // centre filter knobs do nothing when the centre is raw
        crop_one_euro_min_cutoff:   oneEuro && !exprLock,
        crop_one_euro_beta:         oneEuro && !exprLock,
        crop_gaussian_window:       gaussian,
        // crop hardening — meaningless when crop_mode="default" (crop off).
        // safety_margin is honoured by reference_smooth too; containment is
        // not (that mode's box IS the detected box, so it always contains it).
        crop_safety_margin:         cropActive || refSmooth,   // NOT exprLock: forced to 1.0
        crop_containment_check:     cropActive,
        crop_containment_tolerance: cropActive && containOn,
        // separate beta for the size/scale trajectory — only the two modes
        // that filter w/h read it, and only under one_euro.
        crop_size_one_euro_beta:    (jitterless || refSmooth || exprLock) && smoothing === "one_euro",
        auto_smoothing_method:      auto,
        // aspect override is applied while BUILDING a box; reference_smooth
        // keeps the detected aspect by construction and default does no crop.
        preserve_face_aspect:       cropActive,
        // eye-open override sub-options
        eye_open_mode:              eyesOpen,
        eye_open_blink_ear:         eyesOpen && String(get("eye_open_mode")?.value) === "blinks_only",
        // previously ungated
        eye_y_fraction:             eyeAlign === "eye_upper_third",
        au_amplify_neutral_frame:   auAmp > 1.0,
        gaze_kalman_meas_std_deg:   kalmanEngine,
        gaze_kalman_process_std:    kalmanEngine,
        gaze_calibration_frame:     gazeEngine === "iris_geometric",
    };
    for (const w of node.widgets) {
        setHidden(w, (w.name in visible) ? !visible[w.name] : false);
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => {
        const r = orig?.call(w, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "WanAnimateV2.PoseAndFaceDetectionV2.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PoseAndFaceDetectionV2") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            // Every widget that GATES a sibling needs its own callback hook,
            // not just a `visible` entry — otherwise toggling it re-renders
            // nothing until some other widget happens to fire.
            for (const n of [
                "use_clahe",
                "use_blur_for_pose", "use_face_smoothing", "use_constant_face_box",
                "use_iris_smoothing", "iris_smoothing_method", "gaze_lock_eyes",
                "use_blendshape_gaze", "crop_mode", "smoothing_method",
                "crop_containment_check", "auto_smoothing_method",
                "force_eyes_open", "eye_open_mode",
                "gaze_engine", "eye_align_mode", "au_amplify",
            ]) {
                hookWidget(this, n);
            }
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
    },
});
