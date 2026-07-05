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
    const auto       = cropMode === "auto";
    const smoothing  = String(get("smoothing_method")?.value ?? "one_euro");

    // visible[name] = true → show; everything not listed is always shown.
    const visible = {
        blur_radius:                useBlur,
        blur_sigma:                 useBlur,
        // face smoothing / constant-box only matter for crop_mode = auto
        use_face_smoothing:         auto,
        face_smoothing_strength:    auto && faceSmooth,
        use_constant_face_box:      auto,
        face_box_size_px:           auto && constBox,
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
        smoothing_method:           jitterless,
        crop_one_euro_min_cutoff:   jitterless && smoothing === "one_euro",
        crop_one_euro_beta:         jitterless && smoothing === "one_euro",
        crop_gaussian_window:       jitterless && smoothing === "gaussian",
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
            for (const n of [
                "use_blur_for_pose", "use_face_smoothing", "use_constant_face_box",
                "use_iris_smoothing", "iris_smoothing_method", "gaze_lock_eyes",
                "use_blendshape_gaze", "crop_mode", "smoothing_method",
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
