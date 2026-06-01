// pose_gaze_viewer.js — C.2 / C0.6
//
// In-canvas overlay for PoseAndFaceDetectionV2 showing skeleton + iris +
// gaze-arrow per frame, driven by the `ui.viewer_meta` payload emitted
// from PoseAndFaceDetectionV2.process().
//
// Three toggle widgets are added to the node (UI-only — they do not
// affect Python execution):
//   viewer_show_skeleton  BOOLEAN  default true
//   viewer_show_iris      BOOLEAN  default true
//   viewer_show_gaze      BOOLEAN  default true
//
// A frame slider lets you scrub through up to 240 captured frames.
//
// License: Apache-2.0

import { app } from "../../scripts/app.js";
import { C, T, reducedMotion } from "./_c2c_theme.js";
import { reportFailure } from "./_c2c_report.js";

const NODE_CLASS = "PoseAndFaceDetectionV2";
const UI_KEY     = "viewer_meta";

// Compact OpenPose-18 skeleton edges (subset that's always meaningful).
const SKELETON_EDGES = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [0, 15], [14, 16], [15, 17],
];

function parseMeta(message) {
    try {
        const raw = message?.[UI_KEY];
        if (!raw) return null;
        const s = Array.isArray(raw) ? raw[0] : raw;
        return typeof s === "string" ? JSON.parse(s) : s;
    } catch (e) {
        reportFailure("pose_gaze_viewer.parseMeta", e);
        return null;
    }
}

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function drawOverlay(ctx, node, meta) {
    if (!meta || !Array.isArray(meta.frames) || meta.frames.length === 0) {
        ctx.save();
        ctx.font = "11px sans-serif";
        ctx.fillStyle = C.dim;
        ctx.fillText("gaze viewer: no data yet — queue the node",
                     12, node.size[1] - 28);
        ctx.restore();
        return;
    }
    const showSkel = node.__pgv_showSkel?.value !== false;
    const showIris = node.__pgv_showIris?.value !== false;
    const showGaze = node.__pgv_showGaze?.value !== false;
    const frameIdx = clamp(node.__pgv_frame | 0, 0, meta.frames.length - 1);
    const frame = meta.frames[frameIdx];
    if (!frame) return;

    // Reserve a square preview area below the regular widgets.
    const pad = 8;
    const w   = node.size[0] - 2 * pad;
    const previewH = Math.min(220, Math.max(120, w * (meta.src_h / meta.src_w)));
    const x0  = pad;
    const y0  = node.size[1] - previewH - 24;

    ctx.save();
    // Background tile.
    ctx.fillStyle = C.bg3 || C.bg;
    ctx.fillRect(x0, y0, w, previewH);
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, previewH - 1);

    const sx = w / meta.src_w;
    const sy = previewH / meta.src_h;
    const px = (x) => x0 + x * sx;
    const py = (y) => y0 + y * sy;

    // Skeleton.
    if (showSkel && Array.isArray(frame.skeleton)) {
        ctx.strokeStyle = C.blue || "#89b4fa";
        ctx.lineWidth = 2;
        for (const [a, b] of SKELETON_EDGES) {
            const pa = frame.skeleton[a];
            const pb = frame.skeleton[b];
            if (!pa || !pb) continue;
            ctx.beginPath();
            ctx.moveTo(px(pa[0]), py(pa[1]));
            ctx.lineTo(px(pb[0]), py(pb[1]));
            ctx.stroke();
        }
        ctx.fillStyle = C.green || "#a6e3a1";
        for (const p of frame.skeleton) {
            if (!p) continue;
            ctx.beginPath();
            ctx.arc(px(p[0]), py(p[1]), 2.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // Iris dots + gaze arrows.
    const eyes = [
        { key: "right_iris", gkey: "right_gaze", color: C.red || "#f38ba8" },
        { key: "left_iris",  gkey: "left_gaze",  color: C.teal || "#94e2d5" },
    ];
    for (const e of eyes) {
        const ir = frame[e.key];
        const gz = frame[e.gkey];
        if (!ir || ir.length < 2) continue;
        const ix = px(ir[0]);
        const iy = py(ir[1]);
        if (showIris) {
            ctx.fillStyle = e.color;
            ctx.beginPath();
            ctx.arc(ix, iy, 3, 0, Math.PI * 2);
            ctx.fill();
        }
        if (showGaze && gz && gz.length >= 3 && gz[2] > 0.01) {
            const len = 28 * gz[2];
            const ex = ix + gz[0] * len;
            const ey = iy + gz[1] * len;
            ctx.strokeStyle = e.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(ix, iy);
            ctx.lineTo(ex, ey);
            ctx.stroke();
            // arrow head
            const ang = Math.atan2(ey - iy, ex - ix);
            const ah = 5;
            ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - ah * Math.cos(ang - 0.5),
                       ey - ah * Math.sin(ang - 0.5));
            ctx.lineTo(ex - ah * Math.cos(ang + 0.5),
                       ey - ah * Math.sin(ang + 0.5));
            ctx.closePath();
            ctx.fillStyle = e.color;
            ctx.fill();
        }
    }

    // Status pill.
    ctx.font = "10px sans-serif";
    ctx.fillStyle = C.dim || "#888";
    ctx.fillText(
        `frame ${frameIdx + 1}/${meta.frames.length}  engine=${meta.engine || "?"}`,
        x0 + 4, y0 + previewH + 12,
    );
    ctx.restore();
}

app.registerExtension({
    name: "wanv2.pose_gaze_viewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            try {
                this.__pgv_meta  = null;
                this.__pgv_frame = 0;
                this.__pgv_showSkel = this.addWidget("toggle",
                    "viewer_show_skeleton", true, () => { this.setDirtyCanvas(true, false); });
                this.__pgv_showIris = this.addWidget("toggle",
                    "viewer_show_iris", true, () => { this.setDirtyCanvas(true, false); });
                this.__pgv_showGaze = this.addWidget("toggle",
                    "viewer_show_gaze", true, () => { this.setDirtyCanvas(true, false); });
                this.__pgv_slider = this.addWidget("slider",
                    "viewer_frame", 0,
                    (v) => { this.__pgv_frame = v | 0; this.setDirtyCanvas(true, false); },
                    { min: 0, max: 0, step: 1, precision: 0 });
                // Mark these widgets as UI-only (Comfy will skip them on
                // serialize/queue because they have no INPUT counterpart).
                for (const w of [this.__pgv_showSkel, this.__pgv_showIris,
                                 this.__pgv_showGaze, this.__pgv_slider]) {
                    if (w) w.serialize = false;
                }
                // Reserve extra height for the preview.
                if (this.size && this.size[1] < 360) this.size[1] = 360;
            } catch (e) {
                reportFailure("pose_gaze_viewer.onNodeCreated", e);
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            try {
                const meta = parseMeta(message);
                if (meta) {
                    this.__pgv_meta = meta;
                    if (this.__pgv_slider) {
                        this.__pgv_slider.options.max =
                            Math.max(0, (meta.frames?.length || 1) - 1);
                    }
                    this.setDirtyCanvas(true, false);
                }
            } catch (e) {
                reportFailure("pose_gaze_viewer.onExecuted", e);
            }
            return r;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed) return;
            try {
                drawOverlay(ctx, this, this.__pgv_meta);
            } catch (e) {
                reportFailure("pose_gaze_viewer.onDrawForeground", e);
            }
        };

        // Keyboard navigation when the node is the selected node in the
        // LiteGraph canvas. LiteGraph dispatches keydown to the selected
        // node via `onKeyDown(e)`. We step the viewer_frame slider.
        const onKeyDown = nodeType.prototype.onKeyDown;
        nodeType.prototype.onKeyDown = function (ev) {
            // Pass through to any prior handler first.
            const r = onKeyDown?.apply(this, arguments);
            if (r === true) return r;
            try {
                const slider = this.__pgv_slider;
                if (!slider) return r;
                const max = slider.options?.max ?? 0;
                const big = ev.shiftKey ? 10 : 1;
                let next = this.__pgv_frame | 0;
                let handled = false;
                switch (ev.key) {
                    case "ArrowLeft":
                    case "ArrowDown":
                        next = Math.max(0, next - big); handled = true; break;
                    case "ArrowRight":
                    case "ArrowUp":
                        next = Math.min(max, next + big); handled = true; break;
                    case "Home":
                        next = 0; handled = true; break;
                    case "End":
                        next = max; handled = true; break;
                }
                if (handled && next !== this.__pgv_frame) {
                    this.__pgv_frame = next;
                    slider.value = next;
                    if (slider.callback) try { slider.callback(next); } catch (_) {}
                    this.setDirtyCanvas(true, false);
                }
                if (handled && ev.preventDefault) ev.preventDefault();
                return handled || r;
            } catch (e) {
                reportFailure("pose_gaze_viewer.onKeyDown", e);
            }
            return r;
        };
    },
});
