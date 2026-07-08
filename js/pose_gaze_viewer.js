// pose_gaze_viewer.js — always-on Pose · Face · Gaze viewer panel.
//
// REVEALS the detection: a real DOM panel that shows the source frame with
// the pose skeleton, iris dots and gaze arrows drawn ON the image (not on a
// blank tile, and not hidden until you queue). Before a run it shows the
// connected input image with a "queue to detect" hint; after a run it
// overlays the detected skeleton/iris/gaze from `viewer_meta` (which now
// carries downscaled frame previews as the backdrop).
//
// Toggle chips (skeleton / iris / gaze) + a frame scrubber. Read-only for
// now — stage 2 grows this into an editable landmark editor.
//
// License: Apache-2.0

import { app } from "../../scripts/app.js";
import { C } from "./_c2c_theme.js";
import { reportFailure } from "./_c2c_report.js";

const NODE_CLASS = "PoseAndFaceDetectionV2";
const UI_KEY     = "viewer_meta";
const PANEL_H    = 340;

// Compact OpenPose-18 skeleton edges (subset that's always meaningful).
const SKELETON_EDGES = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [0, 15], [14, 16], [15, 17],
];

// Decoded-image cache keyed by data-URL / src (avoid re-decoding each render).
const IMG_CACHE = new Map();
function loadImage(src, onReady) {
    if (!src) return null;
    const hit = IMG_CACHE.get(src);
    if (hit) return hit.complete ? hit : null;
    const img = new Image();
    img.onload = () => { try { onReady?.(); } catch (_) {} };
    img.src = src;
    IMG_CACHE.set(src, img);
    return null;
}

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

// Walk upstream from the node's image input to find a rendered preview so the
// panel can show SOMETHING before the node is queued.
function resolveUpstreamImageURL(node) {
    try {
        const graph = node.graph || app.graph;
        const inp = (node.inputs || []).find(i => /^images?$/i.test(i.name) && i.link != null);
        if (!inp) return null;
        const seen = new Set();
        let queue = [graph.links?.[inp.link]?.origin_id];
        while (queue.length) {
            const id = queue.shift();
            if (id == null || seen.has(id)) continue;
            seen.add(id);
            const n = graph.getNodeById?.(id);
            if (!n) continue;
            if (n.imgs && n.imgs.length && n.imgs[0]?.src) return n.imgs[0].src;
            for (const ni of (n.inputs || [])) {
                if (ni.link != null) queue.push(graph.links?.[ni.link]?.origin_id);
            }
        }
    } catch (_) { /* best-effort */ }
    return null;
}

const _fill = (v, def) => (v || def);

function makePanel(node) {
    const root = document.createElement("div");
    root.className = "pgv-root";
    root.style.cssText = `
        display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
        box-sizing:border-box; font-family:ui-sans-serif,system-ui,sans-serif;
        color:${_fill(C.gray150, "#cdd6f4")};
        background:${_fill(C.scrimDark7, "#181825")}; border-radius:8px; padding:8px;
        overflow:hidden; min-height:0;
    `;

    // Header: title + engine readout + toggle chips.
    const header = document.createElement("div");
    header.style.cssText = "display:flex; align-items:center; gap:6px; flex:0 0 auto; flex-wrap:wrap;";
    const title = document.createElement("span");
    title.textContent = "Pose · Face · Gaze";
    title.style.cssText = "font-weight:600; font-size:12px;";
    const engineEl = document.createElement("span");
    engineEl.style.cssText = `font-size:10.5px; color:${_fill(C.overlay1, "#7f849c")}; margin-left:2px;`;
    const spacer = document.createElement("span");
    spacer.style.cssText = "flex:1 1 auto;";
    header.append(title, engineEl, spacer);

    const state = { skel: true, iris: true, gaze: true };
    const mkChip = (label, key, color) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        const paint = () => {
            b.style.cssText = `
                font-size:10.5px; padding:3px 8px; border-radius:999px; cursor:pointer;
                border:1px solid ${state[key] ? color : _fill(C.surface1, "#45475a")};
                background:${state[key] ? color + "22" : "transparent"};
                color:${state[key] ? color : _fill(C.overlay1, "#7f849c")};
            `;
        };
        paint();
        b.onclick = () => { state[key] = !state[key]; paint(); render(); };
        return b;
    };
    header.append(
        mkChip("🦴 skeleton", "skel", _fill(C.blue, "#89b4fa")),
        mkChip("👁 iris", "iris", _fill(C.red, "#f38ba8")),
        mkChip("↗ gaze", "gaze", _fill(C.teal, "#94e2d5")),
    );
    root.appendChild(header);

    // Fallback banner — reveals when the requested accurate gaze engine
    // silently degraded (e.g. ETH-XGaze checkpoint missing). Hidden otherwise.
    const statusEl = document.createElement("div");
    statusEl.style.cssText = `
        display:none; flex:0 0 auto; font-size:10.5px; line-height:1.35;
        padding:5px 8px; border-radius:6px;
        background:${_fill(C.amber, "#fab387")}1e;
        border:1px solid ${_fill(C.amber, "#fab387")}55;
        color:${_fill(C.amber, "#fab387")};
    `;
    root.appendChild(statusEl);

    // Canvas stage.
    const stage = document.createElement("div");
    stage.style.cssText = `
        position:relative; flex:1 1 auto; min-height:0; border-radius:6px;
        background:${_fill(C.black, "#0b0b12")}; border:1px solid ${_fill(C.surface1, "#45475a")};
        overflow:hidden;
    `;
    const cvs = document.createElement("canvas");
    cvs.style.cssText = "width:100%; height:100%; display:block;";
    stage.appendChild(cvs);
    root.appendChild(stage);

    // Scrubber row.
    const scrubRow = document.createElement("div");
    scrubRow.style.cssText = "display:flex; align-items:center; gap:8px; flex:0 0 auto;";
    const scrub = document.createElement("input");
    scrub.type = "range"; scrub.min = "0"; scrub.max = "0"; scrub.value = "0"; scrub.step = "1";
    scrub.style.cssText = "flex:1 1 auto; accent-color:" + _fill(C.blue, "#89b4fa") + ";";
    const frameEl = document.createElement("span");
    frameEl.style.cssText = `font:11px ui-monospace,monospace; color:${_fill(C.overlay1, "#a6adc8")}; white-space:nowrap;`;
    frameEl.textContent = "frame —";
    scrubRow.append(scrub, frameEl);
    root.appendChild(scrubRow);

    const ctx = cvs.getContext("2d");
    let meta = null;
    let frameIdx = 0;

    function _nearestPreview(idx) {
        const pv = meta?.previews;
        if (!pv || !pv.length) return null;
        let best = pv[0], bd = Infinity;
        for (const p of pv) { const d = Math.abs(p.frame - idx); if (d < bd) { bd = d; best = p; } }
        return best;
    }

    function render() {
        const dpr = window.devicePixelRatio || 1;
        const r = cvs.getBoundingClientRect();
        const W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
        if (cvs.width !== W * dpr || cvs.height !== H * dpr) { cvs.width = W * dpr; cvs.height = H * dpr; }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        // Backdrop: preview for the current frame (after run), else upstream image.
        let bdImg = null, srcW = meta?.src_w || 0, srcH = meta?.src_h || 0;
        const pv = _nearestPreview(frameIdx);
        if (pv) bdImg = loadImage(pv.b64, render);
        if (!bdImg) {
            const up = resolveUpstreamImageURL(node);
            if (up) { bdImg = loadImage(up, render); if (bdImg && !srcW) { srcW = bdImg.naturalWidth; srcH = bdImg.naturalHeight; } }
        }

        // Fit the image (letterbox) → drawn rect (ix,iy,iw,ih).
        let ix = 0, iy = 0, iw = W, ih = H;
        if (bdImg) {
            const iar = bdImg.naturalWidth / bdImg.naturalHeight, sar = W / H;
            if (iar > sar) { iw = W; ih = W / iar; } else { ih = H; iw = H * iar; }
            ix = (W - iw) / 2; iy = (H - ih) / 2;
            ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
            ctx.drawImage(bdImg, ix, iy, iw, ih);
            if (!srcW) { srcW = bdImg.naturalWidth; srcH = bdImg.naturalHeight; }
        } else {
            // Empty state — clear invitation, never a blank void.
            ctx.fillStyle = "rgba(148,158,190,0.5)";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.font = "26px system-ui,sans-serif";
            ctx.fillText("👁", W / 2, H / 2 - 22);
            ctx.fillStyle = "rgba(168,178,208,0.7)";
            ctx.font = "600 12px system-ui,sans-serif";
            ctx.fillText("Connect an image, then Queue", W / 2, H / 2 + 2);
            ctx.fillStyle = "rgba(128,138,166,0.55)";
            ctx.font = "11px system-ui,sans-serif";
            ctx.fillText("the skeleton, iris and gaze appear here on the frame", W / 2, H / 2 + 20);
            ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
        }

        // Overlay (only with detection data). src coords → drawn rect.
        const frame = meta?.frames?.[frameIdx];
        if (frame && srcW && srcH) {
            const sx = iw / srcW, sy = ih / srcH;
            const px = (x) => ix + x * sx, py = (y) => iy + y * sy;

            if (state.skel && Array.isArray(frame.skeleton)) {
                ctx.strokeStyle = _fill(C.blue, "#89b4fa"); ctx.lineWidth = 2;
                for (const [a, b] of SKELETON_EDGES) {
                    const pa = frame.skeleton[a], pb = frame.skeleton[b];
                    if (!pa || !pb) continue;
                    ctx.beginPath(); ctx.moveTo(px(pa[0]), py(pa[1])); ctx.lineTo(px(pb[0]), py(pb[1])); ctx.stroke();
                }
                ctx.fillStyle = _fill(C.green, "#a6e3a1");
                for (const p of frame.skeleton) {
                    if (!p) continue;
                    ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 2.5, 0, Math.PI * 2); ctx.fill();
                }
            }
            const eyes = [
                { key: "right_iris", gkey: "right_gaze", color: _fill(C.red, "#f38ba8") },
                { key: "left_iris",  gkey: "left_gaze",  color: _fill(C.teal, "#94e2d5") },
            ];
            for (const e of eyes) {
                const ir = frame[e.key], gz = frame[e.gkey];
                if (!ir || ir.length < 2) continue;
                const iX = px(ir[0]), iY = py(ir[1]);
                if (state.iris) {
                    ctx.fillStyle = e.color;
                    ctx.beginPath(); ctx.arc(iX, iY, 3.5, 0, Math.PI * 2); ctx.fill();
                }
                if (state.gaze && gz && gz.length >= 3 && gz[2] > 0.01) {
                    const len = 34 * gz[2], ex = iX + gz[0] * len, ey = iY + gz[1] * len;
                    ctx.strokeStyle = e.color; ctx.lineWidth = 2;
                    ctx.beginPath(); ctx.moveTo(iX, iY); ctx.lineTo(ex, ey); ctx.stroke();
                    const ang = Math.atan2(ey - iY, ex - iX), ah = 6;
                    ctx.beginPath(); ctx.moveTo(ex, ey);
                    ctx.lineTo(ex - ah * Math.cos(ang - 0.5), ey - ah * Math.sin(ang - 0.5));
                    ctx.lineTo(ex - ah * Math.cos(ang + 0.5), ey - ah * Math.sin(ang + 0.5));
                    ctx.closePath(); ctx.fillStyle = e.color; ctx.fill();
                }
            }
        } else if (bdImg) {
            // Image present but no detection yet — prompt to queue.
            ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fillRect(ix, iy + ih - 22, iw, 22);
            ctx.fillStyle = "rgba(230,235,245,0.9)"; ctx.font = "11px system-ui,sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Queue to detect skeleton · iris · gaze", ix + iw / 2, iy + ih - 8);
            ctx.textAlign = "left";
        }
    }

    scrub.oninput = () => {
        frameIdx = parseInt(scrub.value, 10) || 0;
        const total = meta?.frames?.length || 0;
        frameEl.textContent = total ? `frame ${frameIdx + 1}/${total}` : "frame —";
        render();
    };

    function setMeta(m) {
        meta = m;
        const total = m?.frames?.length || 0;
        scrub.max = String(Math.max(0, total - 1));
        if (frameIdx > total - 1) frameIdx = 0;
        scrub.value = String(frameIdx);
        frameEl.textContent = total ? `frame ${frameIdx + 1}/${total}` : "frame —";
        // Engine + accuracy readout, honest about what actually ran.
        if (m?.engine) {
            const acc = m.engine_accuracy ? ` · ${m.engine_accuracy}` : "";
            engineEl.textContent = `gaze: ${m.engine}${acc}`;
            engineEl.style.color = m.engine_status
                ? _fill(C.amber, "#fab387") : _fill(C.overlay1, "#7f849c");
        } else {
            engineEl.textContent = "";
        }
        if (m?.engine_status) {
            statusEl.style.display = "block";
            statusEl.textContent = "⚠ " + m.engine_status;
        } else {
            statusEl.style.display = "none";
        }
        // Preload the previews so the backdrop is ready.
        for (const p of (m?.previews || [])) loadImage(p.b64, render);
        render();
    }

    const _ro = new ResizeObserver(() => render());
    _ro.observe(stage);

    return { root, setMeta, render, dispose: () => { try { _ro.disconnect(); } catch (_) {} } };
}

app.registerExtension({
    name: "wanv2.pose_gaze_viewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            try {
                const panel = makePanel(this);
                this.__pgv = panel;
                const host = document.createElement("div");
                host.style.cssText = "width:100%;height:100%;";
                host.appendChild(panel.root);
                const w = this.addDOMWidget("pose_gaze_view", "PGV", host, {
                    serialize: false,
                    getMinHeight: () => PANEL_H,
                    getHeight: () => PANEL_H,
                });
                w.computeSize = () => [this.size?.[0] || 320, PANEL_H];
                if (this.size && this.size[1] < 560) this.setSize([Math.max(this.size[0], 360), Math.max(this.size[1], 560)]);
                // Redraw once the graph settles / an image gets connected.
                setTimeout(() => panel.render(), 60);
                setTimeout(() => panel.render(), 400);
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
                if (meta && this.__pgv) this.__pgv.setMeta(meta);
            } catch (e) {
                reportFailure("pose_gaze_viewer.onExecuted", e);
            }
            return r;
        };

        // Re-render when a connection changes (image just got wired in).
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...a) {
            const r = onConnectionsChange?.apply(this, arguments);
            try { setTimeout(() => this.__pgv?.render(), 50); } catch (_) {}
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this.__pgv?.dispose(); } catch (_) {}
            return onRemoved?.apply(this, arguments);
        };
    },
});
