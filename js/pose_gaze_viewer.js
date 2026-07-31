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
import { ensureC2CKit } from "./_c2c_ui_kit.js";

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
    ensureC2CKit();
    const root = document.createElement("div");
    root.className = "pgv-root c2ck";
    root.style.cssText = `
        display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
        box-sizing:border-box; font-family:ui-sans-serif,system-ui,sans-serif;
        color:#e6e6e6; background:#161616; border-radius:8px; padding:8px;
        overflow:hidden; min-height:0;
    `;

    // Header: title + engine readout pill + toggle chips.
    const header = document.createElement("div");
    header.style.cssText = "display:flex; align-items:center; gap:6px; flex:0 0 auto; flex-wrap:wrap;";
    const title = document.createElement("span");
    title.textContent = "Pose · Face · Gaze";
    title.style.cssText = "font-weight:600; font-size:12px;";
    const engineEl = document.createElement("span");
    engineEl.className = "c2ck-pill";
    engineEl.style.display = "none";
    const spacer = document.createElement("span");
    spacer.style.cssText = "flex:1 1 auto;";
    header.append(title, engineEl, spacer);

    const state = { skel: true, iris: true, gaze: true, edit: false };
    const mkChip = (label, key, color, onToggle) => {
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
        b.onclick = () => { state[key] = !state[key]; paint(); onToggle?.(); render(); };
        b._paint = paint;
        return b;
    };
    header.append(
        mkChip("🦴 skeleton", "skel", _fill(C.blue, "#89b4fa")),
        mkChip("👁 iris", "iris", _fill(C.red, "#f38ba8")),
        mkChip("↗ gaze", "gaze", _fill(C.teal, "#94e2d5")),
    );
    // Edit mode — turns the skeleton into draggable handles that CORRECT the
    // detection. When on, the skeleton chip is forced visible so joints show.
    const editChip = mkChip("✏ edit joints", "edit", _fill(C.mauve, "#cba6f7"), () => {
        if (state.edit && !state.skel) { state.skel = true; skelChip._paint(); }
        resetBtn.style.display = state.edit ? "inline-block" : "none";
        cvs.style.cursor = state.edit ? "crosshair" : "default";
    });
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "↺ reset frame";
    resetBtn.style.cssText = `
        display:none; font-size:10.5px; padding:3px 8px; border-radius:999px; cursor:pointer;
        border:1px solid ${_fill(C.surface1, "#45475a")}; background:transparent;
        color:${_fill(C.overlay1, "#7f849c")};
    `;
    resetBtn.onclick = () => resetFrame();
    const skelChip = header.children[3];   // the "skeleton" chip added above
    header.append(editChip, resetBtn);
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
    let lastMap = null;           // {ix,iy,sx,sy,srcW,srcH} from last render
    let dragJoint = -1;           // joint index being dragged (-1 = none)
    let hoverJoint = -1;          // joint index under the cursor
    let overrides = {};           // {"<frame>": {"<joint>": [x_px, y_px]}}

    const HIT_R = 11;             // grab radius (screen px)

    // --- Manual-correction persistence (writes the node's hidden widget) ---
    function _ovWidget() {
        try { return (node.widgets || []).find(w => w.name === "landmark_overrides_json"); }
        catch (_) { return null; }
    }
    function loadOverrides() {
        try {
            const w = _ovWidget();
            const v = w && w.value;
            overrides = (v && typeof v === "string") ? (JSON.parse(v) || {}) : (v || {});
            if (typeof overrides !== "object" || Array.isArray(overrides)) overrides = {};
        } catch (_) { overrides = {}; }
    }
    function saveOverrides() {
        try {
            const w = _ovWidget();
            if (w) {
                w.value = JSON.stringify(overrides);
                if (typeof w.callback === "function") w.callback(w.value);
            }
            node.setDirtyCanvas?.(true, true);
        } catch (_) { /* best-effort */ }
    }
    function _hasOverride(f, j) {
        const fk = String(f), jk = String(j);
        return !!(overrides[fk] && overrides[fk][jk]);
    }
    function setOverride(f, j, xSrc, ySrc) {
        const fk = String(f), jk = String(j);
        if (!overrides[fk]) overrides[fk] = {};
        overrides[fk][jk] = [Math.round(xSrc), Math.round(ySrc)];
    }
    function resetFrame() {
        const fk = String(frameIdx);
        if (overrides[fk]) { delete overrides[fk]; saveOverrides(); }
        // Note: the displayed skeleton keeps the edited positions until the
        // next Queue re-detects; make that explicit in the hint.
        render();
    }

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
            // Even without a backdrop, letterbox the overlay to the SOURCE
            // aspect ratio so the skeleton isn't stretched (and drag hit-tests
            // stay square). Otherwise it fills the canvas non-uniformly.
            if (srcW && srcH) {
                const iar = srcW / srcH, sar = W / H;
                if (iar > sar) { iw = W; ih = W / iar; } else { ih = H; iw = H * iar; }
                ix = (W - iw) / 2; iy = (H - ih) / 2;
            }
        }

        // Overlay (only with detection data). src coords → drawn rect.
        const frame = meta?.frames?.[frameIdx];
        if (frame && srcW && srcH) {
            const sx = iw / srcW, sy = ih / srcH;
            const px = (x) => ix + x * sx, py = (y) => iy + y * sy;
            lastMap = { ix, iy, sx, sy, srcW, srcH };   // for edit hit-testing

            if (state.skel && Array.isArray(frame.skeleton)) {
                ctx.strokeStyle = _fill(C.blue, "#89b4fa"); ctx.lineWidth = 2;
                for (const [a, b] of SKELETON_EDGES) {
                    const pa = frame.skeleton[a], pb = frame.skeleton[b];
                    if (!pa || !pb) continue;
                    ctx.beginPath(); ctx.moveTo(px(pa[0]), py(pa[1])); ctx.lineTo(px(pb[0]), py(pb[1])); ctx.stroke();
                }
                if (state.edit) {
                    // Grabbable handles: ring + fill, larger; hovered/dragged
                    // joint highlighted; corrected joints tinted mauve.
                    for (let j = 0; j < frame.skeleton.length; j++) {
                        const p = frame.skeleton[j];
                        if (!p) continue;
                        const active = (j === dragJoint || j === hoverJoint);
                        const edited = _hasOverride(frameIdx, j);
                        const cx = px(p[0]), cy = py(p[1]);
                        ctx.beginPath(); ctx.arc(cx, cy, active ? 7 : 5, 0, Math.PI * 2);
                        ctx.fillStyle = edited ? _fill(C.mauve, "#cba6f7") : _fill(C.green, "#a6e3a1");
                        ctx.globalAlpha = active ? 1 : 0.9; ctx.fill(); ctx.globalAlpha = 1;
                        ctx.lineWidth = active ? 2.5 : 1.5;
                        ctx.strokeStyle = active ? _fill(C.text, "#e6e9f0") : "rgba(0,0,0,0.55)";
                        ctx.stroke();
                    }
                } else {
                    ctx.fillStyle = _fill(C.green, "#a6e3a1");
                    for (const p of frame.skeleton) {
                        if (!p) continue;
                        ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 2.5, 0, Math.PI * 2); ctx.fill();
                    }
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
                    // dx/dy now carry the magnitude; fixed gain, no second
                    // multiply by gz[2] or the arrow length is squared.
                    const len = 110, ex = iX + gz[0] * len, ey = iY + gz[1] * len;
                    ctx.strokeStyle = e.color; ctx.lineWidth = 2;
                    ctx.beginPath(); ctx.moveTo(iX, iY); ctx.lineTo(ex, ey); ctx.stroke();
                    const ang = Math.atan2(ey - iY, ex - iX), ah = 6;
                    ctx.beginPath(); ctx.moveTo(ex, ey);
                    ctx.lineTo(ex - ah * Math.cos(ang - 0.5), ey - ah * Math.sin(ang - 0.5));
                    ctx.lineTo(ex - ah * Math.cos(ang + 0.5), ey - ah * Math.sin(ang + 0.5));
                    ctx.closePath(); ctx.fillStyle = e.color; ctx.fill();
                }
            }
            if (state.edit) {
                // Editing hint bar.
                const nEdits = Object.keys(overrides[String(frameIdx)] || {}).length;
                const msg = dragJoint >= 0
                    ? "Release to set · re-Queue bakes it into pose_data"
                    : (nEdits ? `${nEdits} joint${nEdits > 1 ? "s" : ""} corrected — re-Queue to bake`
                              : "Drag a joint to correct it");
                ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(ix, iy + ih - 22, iw, 22);
                ctx.fillStyle = _fill(C.mauve, "#cba6f7"); ctx.font = "11px system-ui,sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(msg, ix + iw / 2, iy + ih - 8);
                ctx.textAlign = "left";
            }
        } else if (bdImg) {
            // Image present but no detection yet — prompt to queue.
            ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fillRect(ix, iy + ih - 22, iw, 22);
            ctx.fillStyle = "rgba(230,235,245,0.9)"; ctx.font = "11px system-ui,sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(state.edit
                ? "Queue once to detect, then drag joints to correct"
                : "Queue to detect skeleton · iris · gaze", ix + iw / 2, iy + ih - 8);
            ctx.textAlign = "left";
        }
    }

    // --- Edit-mode pointer handling (drag skeleton joints to correct) ---
    function _evtSrc(e) {
        if (!lastMap || !lastMap.sx || !lastMap.sy) return null;
        const r = cvs.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        return { sxp: (mx - lastMap.ix) / lastMap.sx, syp: (my - lastMap.iy) / lastMap.sy, mx, my };
    }
    function _pickJoint(mx, my) {
        const frame = meta?.frames?.[frameIdx];
        if (!frame || !Array.isArray(frame.skeleton) || !lastMap) return -1;
        const { ix, iy, sx, sy } = lastMap;
        let best = -1, bd = HIT_R * HIT_R;
        for (let j = 0; j < frame.skeleton.length; j++) {
            const p = frame.skeleton[j];
            if (!p) continue;
            const dx = (ix + p[0] * sx) - mx, dy = (iy + p[1] * sy) - my;
            const d = dx * dx + dy * dy;
            if (d < bd) { bd = d; best = j; }
        }
        return best;
    }
    cvs.addEventListener("pointermove", (e) => {
        if (!state.edit) return;
        const s = _evtSrc(e); if (!s) return;
        if (dragJoint >= 0) {
            const frame = meta.frames[frameIdx];
            const sxp = Math.min(lastMap.srcW, Math.max(0, s.sxp));
            const syp = Math.min(lastMap.srcH, Math.max(0, s.syp));
            frame.skeleton[dragJoint] = [sxp, syp];
            setOverride(frameIdx, dragJoint, sxp, syp);
            e.stopPropagation();
            render();
        } else {
            const h = _pickJoint(s.mx, s.my);
            if (h !== hoverJoint) {
                hoverJoint = h;
                cvs.style.cursor = h >= 0 ? "grab" : "crosshair";
                render();
            }
        }
    });
    cvs.addEventListener("pointerdown", (e) => {
        if (!state.edit) return;
        const s = _evtSrc(e); if (!s) return;
        const j = _pickJoint(s.mx, s.my);
        if (j >= 0) {
            dragJoint = j;
            cvs.style.cursor = "grabbing";
            try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
            e.preventDefault(); e.stopPropagation();
            render();
        }
    });
    function _endDrag(e) {
        if (dragJoint >= 0) {
            dragJoint = -1;
            saveOverrides();
            cvs.style.cursor = state.edit ? "crosshair" : "default";
            try { cvs.releasePointerCapture(e.pointerId); } catch (_) {}
            render();
        }
    }
    cvs.addEventListener("pointerup", _endDrag);
    cvs.addEventListener("pointercancel", _endDrag);

    scrub.oninput = () => {
        frameIdx = parseInt(scrub.value, 10) || 0;
        const total = meta?.frames?.length || 0;
        frameEl.textContent = total ? `frame ${frameIdx + 1}/${total}` : "frame —";
        hoverJoint = -1;
        render();
    };

    function setMeta(m) {
        meta = m;
        loadOverrides();     // resync with the (possibly workflow-loaded) widget
        hoverJoint = -1; dragJoint = -1;
        const total = m?.frames?.length || 0;
        scrub.max = String(Math.max(0, total - 1));
        if (frameIdx > total - 1) frameIdx = 0;
        scrub.value = String(frameIdx);
        frameEl.textContent = total ? `frame ${frameIdx + 1}/${total}` : "frame —";
        // Engine + accuracy readout as a sub-status pill, honest about what ran.
        if (m?.engine) {
            const acc = m.engine_accuracy ? ` · ${m.engine_accuracy}` : "";
            engineEl.textContent = `gaze: ${m.engine}${acc}`;
            engineEl.style.display = "inline-flex";
            engineEl.className = "c2ck-pill " + (m.engine_status ? "off" : "on");
        } else {
            engineEl.style.display = "none";
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

    return { root, setMeta, render, loadOverrides, getMap: () => lastMap, dispose: () => { try { _ro.disconnect(); } catch (_) {} } };
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
                // Hide the raw landmark-overrides widget — it's driven by the
                // editor, not typed by hand (keep its value serialised).
                const ow = (this.widgets || []).find(wd => wd.name === "landmark_overrides_json");
                if (ow) {
                    ow.hidden = true;
                    ow.computeSize = () => [0, -4];
                    ow.type = "hidden";
                }
                // Redraw once the graph settles / an image gets connected.
                setTimeout(() => { panel.loadOverrides?.(); panel.render(); }, 60);
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
