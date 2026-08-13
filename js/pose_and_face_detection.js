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
    // Only expression_lock and default are live. Retired names still
    // present on saved graphs are treated as expression_lock so the old
    // jitterless / auto widget pile does not come back.
    const exprLock   = cropMode !== "default";
    const jitterless = false;
    const refSmooth  = false;
    const auto       = false;
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
    // use_flux only makes sense when a retarget_image is connected (it
    // normalizes the reference pose before retargeting). Hide both flux
    // widgets until one is wired, so the node stays ~12 widgets for the
    // common no-retarget case.
    const hasRetarget = Array.isArray(node.inputs) && node.inputs.some(
        (s) => s && s.name === "retarget_image" && s.link != null);
    const useFlux     = !!get("use_flux")?.value;

    // visible[name] = true → show; everything not listed is always shown.
    const visible = {
        // CLAHE sub-options follow their toggle; the rest of the detector
        // colour controls are always available (they are manual again).
        clahe_clip_limit:           useClahe,
        clahe_grid_size:            useClahe,
        blur_radius:                useBlur,
        blur_sigma:                 useBlur,
        // --- crop geometry -------------------------------------------------
        // Only two modes are offered now (expression_lock, default). Both use
        // the reference face-tight box with a RAW per-frame centre, so every
        // widget that only fed a centre filter, a locked size, or a lag margin
        // is inert and stays hidden. They remain in INPUT_TYPES so existing
        // workflows keep loading with their values intact — widgets are matched
        // by position, and deleting them would shift every later value.
        smoothing_method:           false,
        crop_size_one_euro_beta:    false,
        crop_gaussian_window:       false,
        face_smoothing_strength:    false,
        crop_one_euro_min_cutoff:   oneEuro,
        crop_one_euro_beta:         oneEuro,
        crop_safety_margin:         cropActive || refSmooth,
        crop_containment_check:     cropActive,
        crop_containment_tolerance:  cropActive && containOn,
        preserve_face_aspect:       cropActive,
        auto_smoothing_method:      auto,
        use_face_smoothing:         auto,
        use_constant_face_box:      auto,
        face_box_size_px:           (auto && constBox) || jitterless,
        frame0_cx:                  jitterless,
        frame0_cy:                  jitterless,
        frame0_size:                jitterless,
        keyframes_json:             jitterless,
        // --- gaze / iris ---------------------------------------------------
        // These land in pose_data. The pose conditioning image is a BODY
        // skeleton (draw_aapose_new) with five coarse head dots and no iris,
        // so none of it reaches the model as conditioning — it only populates
        // this node's iris/pupil/debug OUTPUTS.
        //
        // USER CUT (2026-08-13): the gaze stack is HIDDEN entirely now.
        // motion_dim=20 has no gaze channel and the pose image has no iris,
        // so NONE of these widgets can move the rendered eyes — they only
        // feed the iris_data/debug outputs (which the user does not
        // consume) and the pupil-xy outputs (which run on the detector
        // result regardless of these knobs). Hiding them removes the
        // "eye direction going off / yapping" surface area without losing
        // any reachable effect. They stay in INPUT_TYPES so saved workflows
        // keep loading (widgets match by position). To re-expose a knob
        // for experimentation, flip its `false` to a condition here.
        use_iris_smoothing:         false,
        iris_smoothing_strength:    false,
        iris_smoothing_method:      false,
        iris_one_euro_min_cutoff:   false,
        iris_one_euro_beta:         false,
        gaze_lock_eyes:              false,
        gaze_lock_strength:         false,
        use_blendshape_gaze:         false,
        gaze_one_euro_min_cutoff:   false,
        gaze_one_euro_beta:         false,
        gaze_max_yaw_deg:           false,
        gaze_max_pitch_deg:         false,
        gaze_kalman_meas_std_deg:   false,
        gaze_kalman_process_std:    false,
        gaze_calibration_frame:     false,
        gaze_engine:                false,
        // --- pixel edits to face_images (these DO reach the model) ---------
        eye_open_mode:              eyesOpen,
        eye_open_blink_ear:         eyesOpen && String(get("eye_open_mode")?.value) === "blinks_only",
        eye_y_fraction:             eyeAlign === "eye_upper_third",
        au_amplify_neutral_frame:   auAmp > 1.0,
        // --- use_flux (enhanced retargeting) --------------------------------
        // Shown only when retarget_image is connected. use_flux normalizes the
        // reference + first template frame to a front-facing pose via
        // FLUX.1-Kontext-dev before retargeting — only useful for a non-front
        // reference. flux_kontext_path follows the toggle.
        use_flux:                   hasRetarget,
        flux_kontext_path:          hasRetarget && useFlux,
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
                "use_flux",
            ]) {
                hookWidget(this, n);
            }
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        // Re-evaluate visibility when retarget_image is wired/unwired, so
        // the flux widgets appear/disappear with the connection.
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
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
