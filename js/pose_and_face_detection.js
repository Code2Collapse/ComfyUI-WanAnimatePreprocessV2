// PoseAndFaceDetectionV2: HIDE sub-options that don't apply to the current
// toggle/mode selection (user rule 2026-06-24: "show sub-options only when true,
// hide if false"). A sub-option appears only when its controlling boolean is ON
// (or the relevant crop_mode is selected); otherwise it is removed and the node
// shrinks. Re-applied whenever a controlling widget changes.
import { app } from "../../scripts/app.js";
import { setHidden } from "./_widget_visibility.js";

// L2.31: the pack's shared show/hide helper (also hides a DOM widget's wrapper and sets options.hidden) instead of a
// local copy, so both conditional-UI files behave the same.
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
    // L2.33: the Python offers default / expression_lock / jitterless / auto / action again ("the ones you asked to
    // keep"), but this file still treated every non-default mode as expression_lock, so jitterless and auto lost the
    // controls they read. Each mode now shows what its branch in nodes.py uses. Retired names still on saved graphs
    // (central_face, reference_smooth) keep the expression_lock rules.
    const jitterless = cropMode === "jitterless";
    const auto       = cropMode === "auto";
    const action     = cropMode === "action";
    const exprLock   = !["default", "jitterless", "auto", "action"].includes(cropMode);
    const refSmooth  = false;
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
    const faceSr      = String(get("face_sr")?.value ?? "none");
    const gazeUsed    = String(get("apply_gaze_to_face_image")?.value ?? "off") !== "off";

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
        // expression_lock keeps its earlier rules; jitterless / auto / action show what their branch reads
        smoothing_method:           jitterless,
        crop_size_one_euro_beta:    jitterless && method === "one_euro",
        crop_gaussian_window:       (jitterless || (auto && faceSmooth)) && gaussian,
        face_smoothing_strength:    (jitterless && emaLike) || (auto && faceSmooth),
        crop_one_euro_min_cutoff:   oneEuro && (!auto || faceSmooth),
        crop_one_euro_beta:         oneEuro && (!auto || faceSmooth),
        crop_safety_margin:         jitterless || action || (auto && constBox) || refSmooth,
        crop_containment_check:     jitterless || (auto && constBox),
        crop_containment_tolerance:  ((jitterless || (auto && constBox)) && containOn) || action,
        preserve_face_aspect:       jitterless || (auto && constBox),
        auto_smoothing_method:      auto && faceSmooth,
        use_face_smoothing:         auto,
        use_constant_face_box:      auto,
        face_box_size_px:           (auto && constBox) || jitterless,
        frame0_cx:                  jitterless,
        frame0_cy:                  jitterless,
        frame0_size:                jitterless || action,
        keyframes_json:             jitterless,
        // --- gaze / iris ---------------------------------------------------
        // These land in pose_data. The pose conditioning image is a BODY
        // skeleton (draw_aapose_new) with five coarse head dots and no iris,
        // so none of it reaches the model as conditioning — it only populates
        // this node's iris/pupil/debug OUTPUTS.
        //
        // USER CUT (2026-08-13) hid the gaze stack entirely: motion_dim=20 has
        // no gaze channel and the pose image has no iris, so these knobs cannot
        // steer the face code. L2.31 (owner 2026-10-08 "parameters vanish /
        // hidden"): they DO decide the pixels apply_gaze_to_face_image warps
        // into face_images (warp / overlay / replace reach the model), and with
        // that switch on there was no way to tune or even see the engine
        // driving it. So: still hidden while apply_gaze_to_face_image = off
        // (the default - the node stays compact), shown, each behind its own
        // parent toggle, as soon as the warp uses them. They stay in
        // INPUT_TYPES and in this order so saved workflows keep their values.
        use_iris_smoothing:         gazeUsed,
        iris_smoothing_method:      gazeUsed && irisSmooth,
        iris_smoothing_strength:    gazeUsed && irisSmooth && irisMethod === "ema",
        iris_one_euro_min_cutoff:   gazeUsed && irisSmooth && irisMethod === "one_euro",
        iris_one_euro_beta:         gazeUsed && irisSmooth && irisMethod === "one_euro",
        gaze_lock_eyes:             gazeUsed,
        gaze_lock_strength:         gazeUsed && gazeLock,
        use_blendshape_gaze:        gazeUsed,
        gaze_one_euro_min_cutoff:   gazeUsed,
        gaze_one_euro_beta:         gazeUsed,
        gaze_max_yaw_deg:           gazeUsed,
        gaze_max_pitch_deg:         gazeUsed,
        gaze_kalman_meas_std_deg:   gazeUsed && kalmanEngine,
        gaze_kalman_process_std:    gazeUsed && kalmanEngine,
        gaze_calibration_frame:     gazeUsed,
        gaze_engine:                gazeUsed,
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
        // --- face super-resolution (L2.31) --------------------------------
        // face_sr_model is read only by comfy_upscale; the stabiliser by any SR mode.
        face_sr_model:              faceSr === "comfy_upscale",
        face_sr_stabilise:          faceSr !== "none",
    };
    // L2.31: only the widgets this rule set owns. Every other widget is left as it is - re-showing them all here
    // undid pose_gaze_viewer.js hiding landmark_overrides_json (editor-driven, never typed by hand).
    for (const w of node.widgets) {
        if (w.name in visible) setHidden(w, !visible[w.name]);
    }
    // Measured live (L2.31, fe 1.53.10): Nodes 2.0 drops a widget whose type is "hidden" from its rows and takes the
    // node height written here, in both renderers. vueFitNodeHeight made the node GROW on every change (1310 -> 2462
    // px), so it is not used.
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
                "use_flux", "face_sr", "apply_gaze_to_face_image",
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
