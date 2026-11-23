// DrawViTPoseV2: HIDE the iris/gaze sub-parameters when their parent toggle is
// off (user rule 2026-06-24: show sub-options only when true, hide if false).
//   iris_radius / iris_min_confidence / iris_color  -> only when draw_iris
//   gaze_arrow_len                                   -> only when draw_gaze
import { app } from "../../scripts/app.js";
import { setHidden } from "./_widget_visibility.js";

// L2.31: the pack's shared show/hide helper (also hides a DOM widget's wrapper and sets options.hidden) instead of a
// local copy, so both conditional-UI files behave the same.
function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    // face_images and reference_expression_coeffs_json gate their sub-options
    // by CONNECTION, not by a widget value — check the link, not a .value.
    const linked = (n) => (node.inputs || []).find(i => i.name === n)?.link != null;
    const drawIris = !!get("draw_iris")?.value;
    const drawGaze = !!get("draw_gaze")?.value;
    const hasFace  = linked("face_images");
    // reference_expression_coeffs_json is a multiline STRING: it can be
    // wired OR typed, so the critic is active if either is true.
    const refW     = get("reference_expression_coeffs_json");
    const hasRef   = linked("reference_expression_coeffs_json")
                     || String(refW?.value ?? "").trim().length > 0;
    const visible = {
        iris_radius:         drawIris,
        iris_min_confidence: drawIris,
        iris_color:          drawIris,
        gaze_arrow_len:      drawGaze,
        // face passthrough sub-options are meaningless with no face batch.
        // (face_cfg_scale is forceInput now — connection-only, so it is not
        // in node.widgets at all and needs no entry here.)
        enforce_512_face:         hasFace,
        apply_pose_edits_to_face: hasFace,
        // closed-loop critic sub-options only matter once a reference is given
        segment_length:      hasRef,
        top_k_aus:           hasRef,
    };
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
    name: "WanAnimateV2.DrawViTPoseV2.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DrawViTPoseV2") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const n of ["draw_iris", "draw_gaze",
                             "reference_expression_coeffs_json"]) hookWidget(this, n);
            // face_images / reference_* are gated by CONNECTION, so widget
            // callbacks never fire for them — re-evaluate on link changes too.
            const origConn = this.onConnectionsChange;
            this.onConnectionsChange = function (...a) {
                const rr = origConn?.apply(this, a);
                setTimeout(() => applyVisibility(this), 0);
                return rr;
            };
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
