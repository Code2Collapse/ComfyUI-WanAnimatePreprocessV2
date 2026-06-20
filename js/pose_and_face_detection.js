// PoseAndFaceDetectionV2: conditional widget visibility for blur/smoothing/constant-box/iris settings.
import { app } from "../../scripts/app.js";

const ALWAYS = new Set([
    "width","height","detection_threshold","pose_threshold",
    "use_clahe","use_blur_for_pose",
    "use_iris_smoothing","use_mediapipe_face","crop_mode",
]);

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
        if ("__mec_origType" in w) {
            const t = w.__mec_origType;
            if (t === undefined) delete w.type; else w.type = t;
            delete w.__mec_origType;
        }
        if ("__mec_origComputeSize" in w) {
            const cs = w.__mec_origComputeSize;
            if (cs === undefined) delete w.computeSize; else w.computeSize = cs;
            delete w.__mec_origComputeSize;
        }
        // No else: if no original was saved we never hid this widget, so its
        // computeSize is already the legitimate one. Deleting it would destroy
        // DOM-widget sizing (e.g. keyframes_json multiline, viewer_* widgets)
        // on always-visible widgets. (Mirrors the `type` branch above.)
        w.hidden = false;
        const el = w.element;
        if (el) {
            if ("__mec_origElDisplay" in w) {
                const d = w.__mec_origElDisplay;
                el.style.display = d ?? "";
                delete w.__mec_origElDisplay;
            } else {
                el.style.display = "";
            }
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if ("__mec_origWrapDisplay" in w) {
                    const d = w.__mec_origWrapDisplay;
                    wrap.style.display = d ?? "";
                    delete w.__mec_origWrapDisplay;
                } else {
                    wrap.style.display = "";
                }
            }
        }
    }
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const useBlur     = !!get("use_blur_for_pose")?.value;
    const faceSmooth  = !!get("use_face_smoothing")?.value;
    const constBox    = !!get("use_constant_face_box")?.value;
    const irisSmooth  = !!get("use_iris_smoothing")?.value;
    const cropMode    = String(get("crop_mode")?.value ?? "default");
    const jitterless  = cropMode === "jitterless";
    const auto        = cropMode === "auto";
    // "default" = crop off: hide all crop-related controls (smoothing,
    // constant-box size, jitterless widgets) since detected raw bboxes
    // are returned as-is.
    const cropActive  = auto || jitterless;
    const smoothing   = String(get("smoothing_method")?.value ?? "one_euro");
    const conditional = {
        blur_radius:                useBlur,
        blur_sigma:                 useBlur,
        // smoothing/constant-box only matter for "auto"
        use_face_smoothing:         auto,
        face_smoothing_strength:    auto && faceSmooth,
        use_constant_face_box:      auto,
        face_box_size_px:           auto && constBox,
        iris_smoothing_strength:    irisSmooth,
        // jitterless-only widgets
        frame0_cx:                  jitterless,
        frame0_cy:                  jitterless,
        frame0_size:                jitterless,
        keyframes_json:             jitterless,
        smoothing_method:           jitterless,
        crop_one_euro_min_cutoff:   jitterless && smoothing === "one_euro",
        crop_one_euro_beta:         jitterless && smoothing === "one_euro",
        crop_gaussian_window:       jitterless && smoothing === "gaussian",
    };
    void cropActive; // reserved for future widgets
    for (const w of node.widgets) {
        if (ALWAYS.has(w.name)) { setHidden(w, false); continue; }
        if (w.name in conditional) { setHidden(w, !conditional[w.name]); continue; }
        setHidden(w, false);
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
                "use_blur_for_pose","use_face_smoothing","use_constant_face_box",
                "use_iris_smoothing","crop_mode","smoothing_method",
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
