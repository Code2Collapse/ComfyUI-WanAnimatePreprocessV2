// PoseAndFaceDetectionV2: conditional widget visibility for blur/smoothing/constant-box/iris settings.
import { app } from "../../scripts/app.js";

const ALWAYS = new Set([
    "width","height","detection_threshold","pose_threshold",
    "use_clahe","use_blur_for_pose","use_face_smoothing","use_constant_face_box",
    "use_iris_smoothing","use_mediapipe_face",
]);

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (w.__mec_origType === undefined) w.__mec_origType = w.type;
        if (w.__mec_origComputeSize === undefined) w.__mec_origComputeSize = w.computeSize;
        w.computeSize = () => [0, -4];
        w.type = "hidden";
        w.hidden = true;
        const el = w.element;
        if (el) {
            if (w.__mec_origElDisplay === undefined) {
                w.__mec_origElDisplay = el.style.display || "";
            }
            el.style.display = "none";
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if (w.__mec_origWrapDisplay === undefined) {
                    w.__mec_origWrapDisplay = wrap.style.display || "";
                }
                wrap.style.display = "none";
            }
        }
    } else {
        if (w.__mec_origType !== undefined) {
            w.type = w.__mec_origType;
            delete w.__mec_origType;
        }
        if (w.__mec_origComputeSize !== undefined) {
            w.computeSize = w.__mec_origComputeSize;
            delete w.__mec_origComputeSize;
        }
        w.hidden = false;
        const el = w.element;
        if (el) {
            el.style.display = w.__mec_origElDisplay ?? "";
            delete w.__mec_origElDisplay;
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                wrap.style.display = w.__mec_origWrapDisplay ?? "";
                delete w.__mec_origWrapDisplay;
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
    const conditional = {
        blur_radius:              useBlur,
        blur_sigma:               useBlur,
        face_smoothing_strength:  faceSmooth,
        face_box_size_px:         constBox,
        iris_smoothing_strength:  irisSmooth,
    };
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
            for (const n of ["use_blur_for_pose","use_face_smoothing","use_constant_face_box","use_iris_smoothing"]) {
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
