// DrawViTPoseV2: HIDE the iris/gaze sub-parameters when their parent toggle is
// off (user rule 2026-06-24: show sub-options only when true, hide if false).
//   iris_radius / iris_min_confidence / iris_color  -> only when draw_iris
//   gaze_arrow_len                                   -> only when draw_gaze
import { app } from "../../scripts/app.js";

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (!("__c2c_origType" in w)) w.__c2c_origType = w.type;
        if (!("__c2c_origComputeSize" in w)) w.__c2c_origComputeSize = w.computeSize;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.hidden = true;
        const el = w.element;
        if (el) {
            if (!("__c2c_origElDisplay" in w)) w.__c2c_origElDisplay = el.style.display;
            el.style.display = "none";
        }
    } else {
        if ("__c2c_origType" in w) { w.type = w.__c2c_origType; delete w.__c2c_origType; }
        if ("__c2c_origComputeSize" in w) {
            const cs = w.__c2c_origComputeSize;
            if (cs === undefined) delete w.computeSize; else w.computeSize = cs;
            delete w.__c2c_origComputeSize;
        }
        w.hidden = false;
        const el = w.element;
        if (el) { el.style.display = ("__c2c_origElDisplay" in w) ? (w.__c2c_origElDisplay ?? "") : ""; delete w.__c2c_origElDisplay; }
    }
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const drawIris = !!get("draw_iris")?.value;
    const drawGaze = !!get("draw_gaze")?.value;
    const visible = {
        iris_radius:         drawIris,
        iris_min_confidence: drawIris,
        iris_color:          drawIris,
        gaze_arrow_len:      drawGaze,
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
    name: "WanAnimateV2.DrawViTPoseV2.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DrawViTPoseV2") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const n of ["draw_iris", "draw_gaze"]) hookWidget(this, n);
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
