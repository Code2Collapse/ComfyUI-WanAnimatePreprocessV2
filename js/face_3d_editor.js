// SPDX-License-Identifier: Apache-2.0
// face_3d_editor.js — lazy-loaded Three.js editor for WanFaceController3DV2.
//
// Self-contained module. Exports a single function `mount3DEditor(host, opts)`
// that builds a 3D landmark cloud inside `host` (any HTMLElement) and wires
// it to the host node's head_pose widgets via the callbacks in `opts`.
//
// Three.js is loaded lazily from a public CDN with a graceful fallback:
//
//   import("./face_3d_editor.js").then(m => m.mount3DEditor(div, opts))
//                                .catch(err => /* keep 2D view */)
//
// If the CDN load fails (offline, blocked, etc.) the function rejects with
// an Error whose .code is "FACE3D_NO_THREE", and the caller is expected to
// fall back to the 2D canvas view that already exists in face_controller_3d.js.
//
// Public API
// ──────────
//   mount3DEditor(host, {
//       getLandmarks:    () => [[x,y], ... 68 normalised pairs in [0,1]],
//       getHeadPose:     () => ({ yaw, pitch, roll, tx, ty, tz }),
//       setHeadPose:     (partial) => void,    // {tx?, ty?, tz?, yaw?, pitch?, roll?}
//       getFrameLabel:   () => string,          // "frame N / M" for the header
//       theme:           { canvas_bg, text, dim, border, accent, sel },
//       onClose:         () => void,            // called when the user clicks ✕
//   }) → Promise<{ destroy(): void, refresh(): void }>
//
// The returned `refresh()` re-reads landmarks + head pose; call it whenever
// the host node's frame slider moves or external widget changes occur.
//
// All Three.js objects are scoped to the host element; calling destroy()
// disposes geometries, materials, the renderer, and removes every DOM
// node and listener so the editor leaves zero residue.

// ── Canonical iBUG-68 depth table (mirrors python _CANONICAL_Z) ──────
// Convention: +z = AWAY from camera; -z = forward (closer to viewer).
// Values are in face-bbox-normalised units (face_width ≈ 1.0).
const CANONICAL_Z = (() => {
    const z = new Float32Array(68);
    const jaw = [
        +0.18, +0.16, +0.13, +0.10, +0.06, +0.02, -0.02, -0.03,
        -0.04,
        -0.03, -0.02, +0.02, +0.06, +0.10, +0.13, +0.16, +0.18,
    ];
    jaw.forEach((v, i) => { z[i] = v; });
    [-0.04, -0.05, -0.06, -0.06, -0.05].forEach((v, i) => { z[17 + i] = v; });
    [-0.05, -0.06, -0.06, -0.05, -0.04].forEach((v, i) => { z[22 + i] = v; });
    [-0.04, -0.07, -0.10, -0.13].forEach((v, i) => { z[27 + i] = v; });
    [-0.06, -0.09, -0.12, -0.09, -0.06].forEach((v, i) => { z[31 + i] = v; });
    [+0.01, -0.01, -0.02, -0.03, -0.02, -0.01].forEach((v, i) => { z[36 + i] = v; });
    [-0.03, -0.02, -0.01, +0.01, -0.01, -0.02].forEach((v, i) => { z[42 + i] = v; });
    [
        -0.05, -0.06, -0.07, -0.08, -0.07, -0.06,
        -0.05, -0.06, -0.07, -0.08, -0.07, -0.06,
    ].forEach((v, i) => { z[48 + i] = v; });
    [-0.06, -0.07, -0.08, -0.07, -0.06, -0.07, -0.08, -0.07]
        .forEach((v, i) => { z[60 + i] = v; });
    return z;
})();

// iBUG-68 connectivity for the wireframe (jaw + brows + nose + eyes + outer/inner lips).
const IBUG_EDGES = (() => {
    const seqs = [
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],        // jaw
        [17,18,19,20,21],                                    // right brow
        [22,23,24,25,26],                                    // left brow
        [27,28,29,30],                                       // nose bridge
        [31,32,33,34,35],                                    // nose base
        [36,37,38,39,40,41,36],                              // right eye (closed)
        [42,43,44,45,46,47,42],                              // left eye (closed)
        [48,49,50,51,52,53,54,55,56,57,58,59,48],            // outer mouth
        [60,61,62,63,64,65,66,67,60],                        // inner mouth
    ];
    const edges = [];
    for (const s of seqs) {
        for (let i = 0; i + 1 < s.length; i++) edges.push([s[i], s[i + 1]]);
    }
    return edges;
})();

// ── OpenPose-18 body skeleton (mirrors POSE18_NAMES / POSE18_EDGES_DEFAULT
//    in face_controller_3d.js and _POSE18_* in the python node) ─────────
// Joint order: 0 nose,1 neck,2 rShoulder,3 rElbow,4 rWrist,5 lShoulder,
// 6 lElbow,7 lWrist,8 rHip,9 rKnee,10 rAnkle,11 lHip,12 lKnee,13 lAnkle,
// 14 rEye,15 lEye,16 rEar,17 lEar.
const POSE18_EDGES_FALLBACK = [
    [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
    [1,0],[0,14],[14,16],[0,15],[15,17],
];
// Canonical depth profile (+z away from camera, -z toward viewer), small so
// the skeleton starts near-planar; the user drags joints in z to add relief.
const CANONICAL_BODY_Z = new Float32Array([
    -0.05, 0.00, 0.00, -0.03, -0.06, 0.00, -0.03, -0.06,
     0.02, 0.00, -0.02, 0.02, 0.00, -0.02, -0.06, -0.06, 0.00, 0.00,
]);
// Per-limb colours for readability (right side warm, left side cool).
const BODY_JOINT_COLOR = 0x66e0a3;   // default body joint
const BODY_EDGE_COLOR  = 0x9fe9c6;

// Three.js source resolution order:
//   1. Local vendored files alongside this module (three.module.js, three_orbit.js,
//      three_transform.js). This lets the editor work offline.
//   2. CDN fallback (unpkg → jsdelivr). Kept so the editor still works if the
//      user has not vendored yet, but offline + slow networks now succeed.
//
// To vendor (one-time), from this directory:
//   curl -L https://unpkg.com/three@0.158.0/build/three.module.js -o three.module.js
//   curl -L https://unpkg.com/three@0.158.0/examples/jsm/controls/OrbitControls.js -o three_orbit.js
//   curl -L https://unpkg.com/three@0.158.0/examples/jsm/controls/TransformControls.js -o three_transform.js
const THREE_VERSION = "0.158.0";
const CDN_BASES = [
    `https://unpkg.com/three@${THREE_VERSION}`,
    `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}`,
];

async function _tryLocalThreeStack() {
    // import.meta.url lets us locate sibling files without hardcoding the
    // ComfyUI extension path.
    const here = new URL("./", import.meta.url).href;
    const coreUrl = here + "three.module.js";
    // HEAD first so we don't surface a noisy module load error when the file
    // is absent — we want to fall through to CDN silently in that case.
    const head = await fetch(coreUrl, { method: "HEAD" }).catch(() => null);
    if (!head || !head.ok) return null;

    const THREE = await import(/* @vite-ignore */ coreUrl);
    // The local control files now import './three.module.js' (a resolvable
    // relative specifier), so import them directly — no fetch/blob patch needed.
    // This also fixes ComfyUI's own auto-load of these files, which previously
    // errored with "Failed to resolve module specifier 'three'".
    const loadLocalCtrl = async (filename, exportName) => {
        const mod = await import(/* @vite-ignore */ here + filename);
        return mod[exportName];
    };
    const OrbitControls = await loadLocalCtrl("three_orbit.js", "OrbitControls");
    // TransformControls is optional — TransformControls vendoring is a nice-to-have,
    // its absence shouldn't kill the editor since core falls back gracefully.
    let TransformControls = null;
    try { TransformControls = await loadLocalCtrl("three_transform.js", "TransformControls"); }
    catch (_) { TransformControls = null; }
    return { THREE, OrbitControls, TransformControls, cdnBase: "local" };
}

// Module-level cache: the ~1.2 MB three.module.js is parsed ONCE and reused
// on every re-open (closing the editor destroys the DOM, not this promise),
// so the second+ open is instant instead of re-fetching/re-parsing.
let _threeStackPromise = null;
function loadThreeStack() {
    if (!_threeStackPromise) {
        _threeStackPromise = _loadThreeStackUncached().catch((e) => {
            _threeStackPromise = null;   // allow retry after a failure
            throw e;
        });
    }
    return _threeStackPromise;
}

async function _loadThreeStackUncached() {
    // 1. Try vendored copy first.
    try {
        const local = await _tryLocalThreeStack();
        if (local) return local;
    } catch (_) {
        // Local present but failed to load — fall through to CDN.
    }

    // 2. CDN fallback.
    let lastErr = null;
    for (const base of CDN_BASES) {
        try {
            const THREE = await import(/* @vite-ignore */ `${base}/build/three.module.js`);
            const fetchModule = async (path) => {
                const url = `${base}/examples/jsm/${path}`;
                const txt = await fetch(url).then((r) => {
                    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
                    return r.text();
                });
                const patched = txt.replace(
                    /from\s*['"]three['"]/g,
                    `from '${base}/build/three.module.js'`,
                );
                const blob = new Blob([patched], { type: "text/javascript" });
                const blobUrl = URL.createObjectURL(blob);
                try {
                    return await import(/* @vite-ignore */ blobUrl);
                } finally {
                    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
                }
            };
            const orbit = await fetchModule("controls/OrbitControls.js");
            const xform = await fetchModule("controls/TransformControls.js");
            return {
                THREE,
                OrbitControls: orbit.OrbitControls,
                TransformControls: xform.TransformControls,
                cdnBase: base,
            };
        } catch (e) {
            lastErr = e;
        }
    }
    const err = new Error(
        "Three.js load failed (no vendored copy and no CDN reachable): " +
        (lastErr ? lastErr.message : "unknown")
    );
    err.code = "FACE3D_NO_THREE";
    throw err;
}

// Compute a centroid in normalised (xn, yn) → (cx, cy) for the current
// landmark set; used to centre the cloud at origin and scale to ±0.5.
function _centroid(lms) {
    let cx = 0, cy = 0, n = 0;
    for (const p of lms) {
        if (!p) continue;
        cx += p[0]; cy += p[1]; n++;
    }
    if (n === 0) return [0.5, 0.5];
    return [cx / n, cy / n];
}

// Convert iBUG normalised landmarks + canonical Z into a Float32Array of
// xyz triples centred at origin and scaled so 1 unit ≈ face_width.
//
// Input  xn in [0,1] (left→right), yn in [0,1] (top→bottom).
// Output x in [-0.5, +0.5] (right→left in world; flipped for mirror),
//        y in [-0.5, +0.5] (up); z from CANONICAL_Z (depth).
function _landmarksToXYZ(lms) {
    const [cx, cy] = _centroid(lms);
    const out = new Float32Array(68 * 3);
    for (let i = 0; i < 68; i++) {
        const p = lms[i];
        if (!p) {
            // Missing landmark — leave at origin.
            out[i * 3 + 0] = 0; out[i * 3 + 1] = 0; out[i * 3 + 2] = 0;
            continue;
        }
        // Flip x (mirror left/right so the rendered face looks like the user's
        // own face, not the camera's view) and y (image y → world y).
        out[i * 3 + 0] = -(p[0] - cx);
        out[i * 3 + 1] = -(p[1] - cy);
        out[i * 3 + 2] = CANONICAL_Z[i];
    }
    return out;
}

// ── Public entry point ───────────────────────────────────────────────
export async function mount3DEditor(host, opts) {
    if (!host) throw new Error("mount3DEditor: host element required");
    const C = opts.theme || {
        canvas_bg: "#181a1d", text: "#ddd", dim: "#888",
        border: "#333", accent: "#4a9eff", sel: "#ffae42",
    };

    // ── Top-level overlay frame ──────────────────────────────────────
    const WRAP_H = Math.max(220, Number(opts.height) || 340);
    const wrap = document.createElement("div");
    wrap.style.cssText =
        "position:relative;width:100%;height:" + WRAP_H + "px;display:flex;flex-direction:column;" +
        "background:" + C.canvas_bg + ";border:1px solid " + C.border + ";" +
        "border-radius:4px;overflow:hidden;font:11px ui-sans-serif;color:" + C.text + ";";
    host.appendChild(wrap);

    // Header bar with title + close button.
    const hdr = document.createElement("div");
    hdr.style.cssText =
        "display:flex;align-items:center;gap:6px;padding:3px 6px;" +
        "background:" + C.border + ";";
    const title = document.createElement("span");
    title.textContent = "3D head editor";
    title.style.cssText = "font-weight:600;";
    const subtitle = document.createElement("span");
    subtitle.style.cssText = "color:" + C.dim + ";margin-left:4px;";
    subtitle.textContent = "loading 3D engine\u2026";
    // Editing-layer dropdown: choose what the 3D editor poses.
    const layerSel = document.createElement("select");
    layerSel.title = "Editing layer";
    layerSel.style.cssText =
        "margin-left:8px;background:" + C.canvas_bg + ";color:" + C.text + ";" +
        "border:1px solid " + C.border + ";border-radius:3px;padding:1px 4px;" +
        "font:11px ui-sans-serif;cursor:pointer;";
    for (const [v, t] of [
        ["face", "Face / Head"], ["body", "Body skeleton"],
        ["camera", "Frame (pan·zoom·rot)"], ["hands", "Hands (soon)"],
    ]) {
        const o = document.createElement("option");
        o.value = v; o.textContent = t; layerSel.appendChild(o);
    }
    layerSel.value = (opts.getLayer && opts.getLayer()) || "face";
    // "Rotate limb" toggle (body layer only) — FK posing of a limb chain.
    const rotChk = document.createElement("label");
    rotChk.style.cssText =
        "margin-left:8px;display:none;align-items:center;gap:3px;cursor:pointer;" +
        "color:" + C.dim + ";font:11px ui-sans-serif;user-select:none;";
    const rotBox = document.createElement("input");
    rotBox.type = "checkbox";
    rotBox.style.cssText = "accent-color:" + C.accent + ";cursor:pointer;";
    rotChk.append(rotBox, document.createTextNode("Rotate limb"));
    // "Persp" toggle (body layer only) — depth foreshortening in the 2D output.
    const perspChk = document.createElement("label");
    perspChk.title = "Perspective output: a joint's depth (z) foreshortens its 2D position";
    perspChk.style.cssText =
        "margin-left:8px;display:none;align-items:center;gap:3px;cursor:pointer;" +
        "color:" + C.dim + ";font:11px ui-sans-serif;user-select:none;";
    const perspBox = document.createElement("input");
    perspBox.type = "checkbox";
    perspBox.style.cssText = "accent-color:" + C.accent + ";cursor:pointer;";
    perspChk.append(perspBox, document.createTextNode("Persp"));
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u2715";
    closeBtn.title = "Close 3D editor";
    closeBtn.style.cssText =
        "margin-left:auto;background:transparent;color:" + C.text + ";" +
        "border:1px solid " + C.border + ";border-radius:3px;" +
        "padding:1px 6px;cursor:pointer;";
    hdr.append(title, subtitle, layerSel, rotChk, perspChk, closeBtn);
    wrap.appendChild(hdr);
    rotBox.addEventListener("change", () => { rotateLimb = rotBox.checked; });
    perspBox.addEventListener("change", () => { outputPersp = perspBox.checked; _reprojectAll(); });

    // Loading placeholder (shown until Three.js resolves or fails).
    const loadingMsg = document.createElement("div");
    loadingMsg.style.cssText =
        "flex:1;display:flex;align-items:center;justify-content:center;color:" + C.dim + ";";
    loadingMsg.textContent = "loading 3D engine\u2026";
    wrap.appendChild(loadingMsg);

    let destroyed = false;
    const cleanups = [];
    const destroy = () => {
        if (destroyed) return;
        destroyed = true;
        for (const fn of cleanups.splice(0)) {
            try { fn(); } catch (_) { /* ignore */ }
        }
        try { host.removeChild(wrap); } catch (_) { /* host may have changed */ }
    };
    closeBtn.addEventListener("click", () => {
        try { opts.onClose?.(); } catch (_) { /* ignore */ }
        destroy();
    });
    cleanups.push(() => closeBtn.replaceWith(closeBtn.cloneNode(true)));

    // Kick off the lazy load. We attach the rest of the UI only after
    // Three.js resolves so the user sees a clear progress message.
    let stack;
    try {
        stack = await loadThreeStack();
    } catch (err) {
        loadingMsg.style.color = "#ff7070";
        loadingMsg.textContent = "Three.js CDN load failed (" + (err?.message || err) +
                                 "). 3D view unavailable; the 2D canvas above still works.";
        // Return a stub so the caller can still call destroy()/refresh().
        return {
            destroy,
            refresh: () => { /* no-op */ },
            failed: true,
            error: err,
        };
    }
    if (destroyed) return { destroy, refresh: () => {} };

    const { THREE, OrbitControls, TransformControls, cdnBase } = stack;
    subtitle.textContent = cdnBase === "local" ? "ready \u00b7 drag to orbit" : ("three@" + THREE_VERSION + " \u00b7 " + cdnBase.replace(/^https?:\/\//, ""));
    loadingMsg.remove();

    // Active editing layer ("face" | "body" | "hands" | "camera").
    let currentLayer = layerSel.value || "face";
    let rotateLimb = false;   // body layer: rotate a limb (FK) vs. free-drag a joint

    // ── Three.js scene setup ─────────────────────────────────────────
    const sceneHost = document.createElement("div");
    sceneHost.style.cssText = "flex:1;position:relative;min-height:0;";
    wrap.appendChild(sceneHost);

    // clientHeight is 0 at mount (flex parent not laid out yet) — derive the
    // initial size from the known wrap height minus the header; the
    // ResizeObserver below corrects it once layout settles. Relying on
    // clientHeight alone created a 0-tall canvas (the "3D vanishes" bug).
    const W0 = Math.max(160, sceneHost.clientWidth  || wrap.clientWidth || host.clientWidth || 320);
    const H0 = Math.max(160, sceneHost.clientHeight || (WRAP_H - 34));

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.canvas_bg);

    const camera = new THREE.PerspectiveCamera(40, W0 / H0, 0.01, 100);
    camera.position.set(0, 0, 2.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(W0, H0, false);
    renderer.domElement.style.cssText = "display:block;width:100%;height:100%;";
    sceneHost.appendChild(renderer.domElement);
    cleanups.push(() => {
        try { renderer.dispose(); } catch (_) {}
        try { renderer.forceContextLoss(); } catch (_) {}
        try { renderer.domElement.remove(); } catch (_) {}
    });

    // Lights — soft ambient + one directional from front-right.
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.95);
    dir.position.set(1, 1, 2);
    scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.35);
    dir2.position.set(-1, 0.5, -1);
    scene.add(dir2);

    // Reference axes (subtle, near origin) for orientation cues.
    const axes = new THREE.AxesHelper(0.25);
    axes.material.transparent = true;
    axes.material.opacity = 0.35;
    scene.add(axes);

    // ── Head group: contains the landmark cloud + edges. yaw/pitch/roll
    //    are applied as this group's Euler rotation; tx/ty/tz as its
    //    position. TransformControls is attached to this group so the
    //    user can drag-translate the whole head.
    const headGroup = new THREE.Group();
    scene.add(headGroup);

    // Landmark spheres (instanced for cheap 68-point cloud).
    const sphereGeom = new THREE.SphereGeometry(0.012, 10, 8);
    const baseMat   = new THREE.MeshLambertMaterial({ color: 0x4a9eff });
    const hoverMat  = new THREE.MeshLambertMaterial({ color: 0xffae42 });
    const meshes = [];
    for (let i = 0; i < 68; i++) {
        const m = new THREE.Mesh(sphereGeom, baseMat);
        m.userData.lmIndex = i;
        headGroup.add(m);
        meshes.push(m);
    }
    cleanups.push(() => {
        sphereGeom.dispose();
        baseMat.dispose();
        hoverMat.dispose();
    });

    // Edge wireframe.
    const edgeGeom = new THREE.BufferGeometry();
    const edgePos = new Float32Array(IBUG_EDGES.length * 2 * 3);
    edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
    const edgeMat = new THREE.LineBasicMaterial({
        color: 0x99c4ff, transparent: true, opacity: 0.55,
    });
    const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
    headGroup.add(edgeLines);
    cleanups.push(() => { edgeGeom.dispose(); edgeMat.dispose(); });

    // ── Body skeleton group (OpenPose-18) ────────────────────────────
    const bodyEdges = (Array.isArray(opts.bodyEdges) && opts.bodyEdges.length)
        ? opts.bodyEdges : POSE18_EDGES_FALLBACK;
    const bodyNames = Array.isArray(opts.bodyNames) ? opts.bodyNames : [];
    const bodyGroup = new THREE.Group();
    bodyGroup.visible = false;
    scene.add(bodyGroup);
    const bodyJointGeom = new THREE.SphereGeometry(0.02, 12, 10);
    const bodyMat = new THREE.MeshLambertMaterial({ color: BODY_JOINT_COLOR });
    const bodyHoverMat = new THREE.MeshLambertMaterial({ color: 0xffae42 });
    const bodyMeshes = [];
    for (let i = 0; i < 18; i++) {
        const m = new THREE.Mesh(bodyJointGeom, bodyMat);
        m.userData.bodyIndex = i;
        m.visible = false;
        bodyGroup.add(m);
        bodyMeshes.push(m);
    }
    const bodyEdgeGeom = new THREE.BufferGeometry();
    const bodyEdgePos = new Float32Array(bodyEdges.length * 2 * 3);
    bodyEdgeGeom.setAttribute("position", new THREE.BufferAttribute(bodyEdgePos, 3));
    const bodyEdgeMat = new THREE.LineBasicMaterial({
        color: BODY_EDGE_COLOR, transparent: true, opacity: 0.7,
    });
    const bodyEdgeLines = new THREE.LineSegments(bodyEdgeGeom, bodyEdgeMat);
    bodyGroup.add(bodyEdgeLines);
    cleanups.push(() => {
        bodyJointGeom.dispose(); bodyMat.dispose(); bodyHoverMat.dispose();
        bodyEdgeGeom.dispose(); bodyEdgeMat.dispose();
    });

    // World<->image mapping for the 2D pose output. WYSIWYG, NOT mirrored (unlike
    // the face cloud) so "drag right" moves the joint right both here AND in the
    // 2D OpenPose fed to Wan; only Y is flipped (image Y down, world Y up).
    //
    // Two output projections (toggle in the header — default Ortho = the verified
    // behavior, zero regression):
    //   Ortho  : imgX = world.x+0.5,            imgY = 0.5-world.y     (depth ignored)
    //   Persp  : s = camZ/(camZ+world.z);  imgX = 0.5+world.x*s, imgY = 0.5-world.y*s
    //            → joints pushed back (z>0) foreshorten toward centre, pulled
    //              forward (z<0) enlarge. Depth finally affects the 2D pose.
    const OUT_CAM_Z = 2.4;          // output camera distance (front)
    let outputPersp = false;        // false = orthographic (default)
    let selJoint = -1;              // last-picked body joint (for the depth slider)
    const _clamp01 = (v) => Math.max(0, Math.min(1, v));
    const _perspScale = (wz) => (outputPersp ? OUT_CAM_Z / (OUT_CAM_Z + (wz || 0)) : 1);
    const bodyXYZ = new Float32Array(18 * 3);
    const bodyValid = new Array(18).fill(false);
    const _imgToWorld = (x, y, wz) => { const s = _perspScale(wz); return [(x - 0.5) / s, -(y - 0.5) / s]; };
    const _worldToImg = (wx, wy, wz) => { const s = _perspScale(wz); return [_clamp01(0.5 + wx * s), _clamp01(0.5 - wy * s)]; };
    function _seedBody() {
        let joints = null;
        try { joints = opts.getBodyJoints?.(); } catch (_) {}
        for (let i = 0; i < 18; i++) {
            const p = Array.isArray(joints) ? joints[i] : null;
            if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) {
                const z = CANONICAL_BODY_Z[i] || 0;
                const [wx, wy] = _imgToWorld(p[0], p[1], z);
                bodyXYZ[i*3] = wx; bodyXYZ[i*3+1] = wy;
                bodyXYZ[i*3+2] = z;
                bodyValid[i] = true;
            } else {
                bodyValid[i] = false;
            }
        }
        _syncBodyMeshes();
    }
    function _syncBodyMeshes() {
        for (let i = 0; i < 18; i++) {
            bodyMeshes[i].visible = bodyValid[i] && bodyGroup.visible;
            if (bodyValid[i]) {
                bodyMeshes[i].position.set(bodyXYZ[i*3], bodyXYZ[i*3+1], bodyXYZ[i*3+2]);
            }
        }
        let w = 0;
        for (const [a, b] of bodyEdges) {
            if (!bodyValid[a] || !bodyValid[b]) { for (let k = 0; k < 6; k++) bodyEdgePos[w++] = 0; continue; }
            bodyEdgePos[w++] = bodyXYZ[a*3]; bodyEdgePos[w++] = bodyXYZ[a*3+1]; bodyEdgePos[w++] = bodyXYZ[a*3+2];
            bodyEdgePos[w++] = bodyXYZ[b*3]; bodyEdgePos[w++] = bodyXYZ[b*3+1]; bodyEdgePos[w++] = bodyXYZ[b*3+2];
        }
        bodyEdgeGeom.attributes.position.needsUpdate = true;
    }

    // FK skeleton for "Rotate limb" mode (neck=root). Dragging a joint rotates
    // the bone parent→joint and carries its whole subtree, keeping bone lengths.
    const BODY_PARENT = {
        0:1, 14:0, 15:0, 16:14, 17:15,
        2:1, 3:2, 4:3, 5:1, 6:5, 7:6,
        8:1, 9:8, 10:9, 11:1, 12:11, 13:12,
    };
    const BODY_CHILDREN = (() => {
        const c = {}; for (let i = 0; i < 18; i++) c[i] = [];
        for (const k in BODY_PARENT) c[BODY_PARENT[k]].push(+k);
        return c;
    })();
    function _subtree(j) {
        const out = [], stack = [j];
        while (stack.length) { const x = stack.pop(); out.push(x); for (const ch of (BODY_CHILDREN[x] || [])) stack.push(ch); }
        return out;
    }
    const _qRot = new THREE.Quaternion();
    const _vOld = new THREE.Vector3(), _vNew = new THREE.Vector3();
    const _vPivot = new THREE.Vector3(), _vTmp = new THREE.Vector3();
    function _applyLimbRotate(j, nx, ny, nz) {
        const p = BODY_PARENT[j];
        if (p === undefined || !bodyValid[p] || !bodyValid[j]) return false;
        _vPivot.set(bodyXYZ[p*3], bodyXYZ[p*3+1], bodyXYZ[p*3+2]);
        _vOld.set(bodyXYZ[j*3] - _vPivot.x, bodyXYZ[j*3+1] - _vPivot.y, bodyXYZ[j*3+2] - _vPivot.z);
        _vNew.set(nx - _vPivot.x, ny - _vPivot.y, nz - _vPivot.z);
        if (_vOld.lengthSq() < 1e-9 || _vNew.lengthSq() < 1e-9) return false;
        _qRot.setFromUnitVectors(_vOld.clone().normalize(), _vNew.clone().normalize());
        const map = {};
        for (const idx of _subtree(j)) {
            if (!bodyValid[idx]) continue;
            _vTmp.set(bodyXYZ[idx*3] - _vPivot.x, bodyXYZ[idx*3+1] - _vPivot.y, bodyXYZ[idx*3+2] - _vPivot.z);
            _vTmp.applyQuaternion(_qRot).add(_vPivot);
            bodyXYZ[idx*3] = _vTmp.x; bodyXYZ[idx*3+1] = _vTmp.y; bodyXYZ[idx*3+2] = _vTmp.z;
            map[idx] = _worldToImg(_vTmp.x, _vTmp.y, _vTmp.z);
        }
        try { opts.setBodyJoints?.(map); } catch (_) {}
        return true;
    }

    // ── Controls ─────────────────────────────────────────────────────
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.1;
    orbit.minDistance = 0.6;
    orbit.maxDistance = 6.0;
    orbit.target.set(0, 0, 0);
    cleanups.push(() => orbit.dispose());

    const xform = new TransformControls(camera, renderer.domElement);
    xform.setMode("translate");
    xform.setSize(0.6);
    xform.attach(headGroup);
    scene.add(xform);
    // Disable orbit while dragging the gizmo (standard Three.js pattern).
    xform.addEventListener("dragging-changed", (ev) => {
        orbit.enabled = !ev.value;
    });
    xform.addEventListener("objectChange", () => {
        // Push the new translation back to the host node's tx/ty/tz widgets.
        try {
            opts.setHeadPose?.({
                tx: headGroup.position.x,
                ty: headGroup.position.y,
                tz: headGroup.position.z,
            });
        } catch (_) { /* widget setter may throw on unmount */ }
    });
    cleanups.push(() => {
        try { xform.detach(); } catch (_) {}
        try { xform.dispose(); } catch (_) {}
    });

    // ── Raycaster picking ────────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    const pickMouse = new THREE.Vector2();
    let hoverIdx = -1;
    const _setHover = (idx) => {
        if (idx === hoverIdx) return;
        if (hoverIdx >= 0) meshes[hoverIdx].material = baseMat;
        hoverIdx = idx;
        if (hoverIdx >= 0) meshes[hoverIdx].material = hoverMat;
        hoverLabel.textContent = (hoverIdx >= 0)
            ? `landmark ${hoverIdx}`
            : "hover a landmark";
    };
    const onMouseMove = (ev) => {
        if (currentLayer !== "face") return;   // body layer has its own picker
        const rect = renderer.domElement.getBoundingClientRect();
        pickMouse.x = ((ev.clientX - rect.left) / rect.width)  * 2 - 1;
        pickMouse.y = -((ev.clientY - rect.top)  / rect.height) * 2 + 1;
        raycaster.setFromCamera(pickMouse, camera);
        const hits = raycaster.intersectObjects(meshes, false);
        _setHover(hits.length > 0 ? hits[0].object.userData.lmIndex : -1);
    };
    renderer.domElement.addEventListener("mousemove", onMouseMove);
    cleanups.push(() => renderer.domElement.removeEventListener("mousemove", onMouseMove));

    // ── HUD: yaw/pitch/roll sliders + readout strip ──────────────────
    const hud = document.createElement("div");
    hud.style.cssText =
        "padding:4px 6px;background:" + C.border + ";border-top:1px solid " + C.border + ";" +
        "display:grid;grid-template-columns:auto 1fr auto;gap:3px 6px;align-items:center;";
    wrap.appendChild(hud);

    const _mkSlider = (label, min, max, step, initial, setter) => {
        const lab = document.createElement("span");
        lab.textContent = label;
        const sl = document.createElement("input");
        sl.type = "range";
        sl.min = String(min); sl.max = String(max); sl.step = String(step);
        sl.value = String(initial);
        sl.style.cssText = "width:100%;accent-color:" + C.accent + ";";
        const val = document.createElement("span");
        val.style.cssText = "font:11px ui-monospace,monospace;color:" + C.dim + ";min-width:48px;text-align:right;";
        val.textContent = Number(initial).toFixed(1);
        sl.addEventListener("input", () => {
            const v = parseFloat(sl.value);
            val.textContent = v.toFixed(1);
            setter(v);
        });
        hud.append(lab, sl, val);
        return { slider: sl, val };
    };

    const headState = opts.getHeadPose?.() || { yaw: 0, pitch: 0, roll: 0, tx: 0, ty: 0, tz: 0 };
    const yawCtl = _mkSlider("yaw\u00b0",   -60, 60, 0.5, headState.yaw   || 0, (v) => {
        _applyEuler({ yaw: v });
        opts.setHeadPose?.({ yaw: v });
    });
    const pitchCtl = _mkSlider("pitch\u00b0", -45, 45, 0.5, headState.pitch || 0, (v) => {
        _applyEuler({ pitch: v });
        opts.setHeadPose?.({ pitch: v });
    });
    const rollCtl = _mkSlider("roll\u00b0",  -45, 45, 0.5, headState.roll  || 0, (v) => {
        _applyEuler({ roll: v });
        opts.setHeadPose?.({ roll: v });
    });

    const hoverLabel = document.createElement("div");
    hoverLabel.style.cssText =
        "padding:2px 6px;background:" + C.canvas_bg + ";color:" + C.dim + ";" +
        "font:11px ui-monospace,monospace;border-top:1px solid " + C.border + ";";
    hoverLabel.textContent = "hover a landmark";
    wrap.appendChild(hoverLabel);

    // ── Framing ("camera") controls: pan / zoom / rotate the whole figure
    //    in the output frame. Works in image space, writes ALL body joints to
    //    pose_overrides_json (so the figure is repositioned in the 2D pose Wan
    //    receives). A true 3D output camera arrives with perspective (Slice 4).
    const camBar = document.createElement("div");
    camBar.style.cssText =
        "display:none;padding:4px 6px;background:" + C.border + ";" +
        "grid-template-columns:auto 1fr auto;gap:3px 6px;align-items:center;";
    wrap.appendChild(camBar);
    const _mkCamSlider = (label, min, max, step, init) => {
        const lab = document.createElement("span"); lab.textContent = label;
        const sl = document.createElement("input"); sl.type = "range";
        sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(init);
        sl.style.cssText = "width:100%;accent-color:" + C.accent + ";";
        const val = document.createElement("span");
        val.style.cssText = "font:11px ui-monospace,monospace;color:" + C.dim + ";min-width:46px;text-align:right;";
        val.textContent = Number(init).toFixed(2);
        sl.addEventListener("input", () => { val.textContent = Number(sl.value).toFixed(2); _applyFraming(); });
        camBar.append(lab, sl, val);
        return sl;
    };
    const panXSl = _mkCamSlider("pan X", -0.5, 0.5, 0.005, 0);
    const panYSl = _mkCamSlider("pan Y", -0.5, 0.5, 0.005, 0);
    const zoomSl = _mkCamSlider("zoom", 0.3, 2.0, 0.01, 1);
    const rotSl  = _mkCamSlider("rotate°", -180, 180, 1, 0);
    const camReset = document.createElement("button");
    camReset.textContent = "Reset framing";
    camReset.style.cssText =
        "grid-column:1 / -1;justify-self:start;background:transparent;color:" + C.text +
        ";border:1px solid " + C.border + ";border-radius:3px;padding:1px 8px;cursor:pointer;";
    camReset.addEventListener("click", () => {
        panXSl.value = "0"; panYSl.value = "0"; zoomSl.value = "1"; rotSl.value = "0";
        camBar.querySelectorAll("span:nth-child(3n)").forEach(() => {});
        _captureCamBase(); _applyFraming();
    });
    camBar.appendChild(camReset);
    let _camBase = null, _camCx = 0.5, _camCy = 0.5;
    function _captureCamBase() {
        let j = null; try { j = opts.getBodyJoints?.(); } catch (_) {}
        _camBase = new Array(18).fill(null);
        let sx = 0, sy = 0, nc = 0;
        for (let i = 0; i < 18; i++) {
            const p = Array.isArray(j) ? j[i] : null;
            if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) { _camBase[i] = [p[0], p[1]]; sx += p[0]; sy += p[1]; nc++; }
        }
        _camCx = nc ? sx / nc : 0.5; _camCy = nc ? sy / nc : 0.5;
    }
    function _applyFraming() {
        if (!_camBase) return;
        const panX = +panXSl.value, panY = +panYSl.value, zoom = +zoomSl.value, rot = (+rotSl.value) * Math.PI / 180;
        const cos = Math.cos(rot), sin = Math.sin(rot), map = {};
        for (let i = 0; i < 18; i++) {
            const b = _camBase[i]; if (!b) continue;
            const dx = b[0] - _camCx, dy = b[1] - _camCy;
            const rx = dx * cos - dy * sin, ry = dx * sin + dy * cos;
            const nx = Math.max(0, Math.min(1, _camCx + rx * zoom + panX));
            const ny = Math.max(0, Math.min(1, _camCy + ry * zoom + panY));
            map[i] = [nx, ny];
            const [wx, wy] = _imgToWorld(nx, ny, bodyXYZ[i*3+2]); bodyXYZ[i*3] = wx; bodyXYZ[i*3+1] = wy;
        }
        _syncBodyMeshes();
        try { opts.setBodyJoints?.(map); } catch (_) {}
    }

    // Re-project every body joint through the current output projection and
    // write the result (used when toggling Persp or resetting depths).
    function _reprojectAll() {
        const map = {};
        for (let i = 0; i < 18; i++) {
            if (!bodyValid[i]) continue;
            map[i] = _worldToImg(bodyXYZ[i*3], bodyXYZ[i*3+1], bodyXYZ[i*3+2]);
        }
        _syncBodyMeshes();
        try { opts.setBodyJoints?.(map); } catch (_) {}
    }

    // ── Body controls bar: depth (z) of the selected joint + reset ───
    const bodyBar = document.createElement("div");
    bodyBar.style.cssText =
        "display:none;padding:4px 6px;background:" + C.border + ";" +
        "grid-template-columns:auto 1fr auto;gap:3px 6px;align-items:center;";
    wrap.appendChild(bodyBar);
    const dLab = document.createElement("span"); dLab.textContent = "depth (sel)";
    const depthSl = document.createElement("input");
    depthSl.type = "range"; depthSl.min = "-0.6"; depthSl.max = "0.6"; depthSl.step = "0.01"; depthSl.value = "0";
    depthSl.style.cssText = "width:100%;accent-color:" + C.accent + ";";
    const depthVal = document.createElement("span");
    depthVal.style.cssText = "font:11px ui-monospace,monospace;color:" + C.dim + ";min-width:46px;text-align:right;";
    depthVal.textContent = "0.00";
    depthSl.addEventListener("input", () => {
        depthVal.textContent = Number(depthSl.value).toFixed(2);
        if (selJoint >= 0 && bodyValid[selJoint]) {
            bodyXYZ[selJoint*3+2] = +depthSl.value;
            _syncBodyMeshes();
            const m = {}; m[selJoint] = _worldToImg(bodyXYZ[selJoint*3], bodyXYZ[selJoint*3+1], bodyXYZ[selJoint*3+2]);
            try { opts.setBodyJoints?.(m); } catch (_) {}
        }
    });
    bodyBar.append(dLab, depthSl, depthVal);
    const depthReset = document.createElement("button");
    depthReset.textContent = "Reset depths";
    depthReset.style.cssText =
        "grid-column:1 / -1;justify-self:start;background:transparent;color:" + C.text +
        ";border:1px solid " + C.border + ";border-radius:3px;padding:1px 8px;cursor:pointer;";
    depthReset.addEventListener("click", () => {
        for (let i = 0; i < 18; i++) if (bodyValid[i]) bodyXYZ[i*3+2] = CANONICAL_BODY_Z[i] || 0;
        if (selJoint >= 0) { depthSl.value = String(bodyXYZ[selJoint*3+2] || 0); depthVal.textContent = Number(bodyXYZ[selJoint*3+2] || 0).toFixed(2); }
        _reprojectAll();
    });
    bodyBar.appendChild(depthReset);

    // ── Editing-layer switching ──────────────────────────────────────
    function setLayer(layer) {
        currentLayer = layer;
        const isFace = layer === "face";
        const isBody = layer === "body";
        const isCamera = layer === "camera";
        headGroup.visible = isFace;
        edgeLines.visible = isFace;
        for (const m of meshes) m.visible = isFace;
        bodyGroup.visible = isBody || isCamera;   // show skeleton while framing too
        // The head-translate gizmo + yaw/pitch/roll sliders apply to the face only.
        try {
            if (isFace) { xform.attach(headGroup); xform.visible = true; xform.enabled = true; }
            else { xform.detach(); xform.visible = false; xform.enabled = false; }
        } catch (_) {}
        hud.style.display = isFace ? "" : "none";
        rotChk.style.display = isBody ? "inline-flex" : "none";
        perspChk.style.display = isBody ? "inline-flex" : "none";
        bodyBar.style.display = isBody ? "grid" : "none";
        camBar.style.display = isCamera ? "grid" : "none";
        if (isBody || isCamera) _seedBody(); else _syncBodyMeshes();
        if (isCamera) _captureCamBase();
        title.textContent = isBody ? "3D body editor"
            : isFace ? "3D head editor"
            : isCamera ? "Framing · pan / zoom / rotate"
            : "3D hands (coming soon)";
        if (isBody) hoverLabel.textContent = "drag a body joint · right-click resets it";
        else if (isCamera) hoverLabel.textContent = "pan / zoom / rotate the whole figure in frame";
        else if (layer === "hands") hoverLabel.textContent = "hands editor needs the hand-keypoint channel — coming next";
        else hoverLabel.textContent = "hover a landmark";
        try { opts.onLayerChange?.(layer); } catch (_) {}
    }
    layerSel.addEventListener("change", () => setLayer(layerSel.value));
    cleanups.push(() => layerSel.replaceWith(layerSel.cloneNode(true)));

    // ── Body-joint picking + 3D drag (projects to 2D pose_overrides) ──
    const _dragNDC = new THREE.Vector2();
    const _dragPoint = new THREE.Vector3();
    const _camDir = new THREE.Vector3();
    // Grab state: a drag plane fixed at mousedown + the offset between the joint
    // and the exact point grabbed, so the joint tracks the pointer with no jump
    // and no axis inversion ("what you grab is what moves, where you point").
    const _grabPlane = new THREE.Plane();
    const _grabHit = new THREE.Vector3();
    const _grabOffset = new THREE.Vector3();
    let bodyDrag = -1;
    const _setNDC = (ev) => {
        const rect = renderer.domElement.getBoundingClientRect();
        _dragNDC.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        _dragNDC.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    };
    function _bodyPick(ev) {
        _setNDC(ev);
        raycaster.setFromCamera(_dragNDC, camera);
        const targets = bodyMeshes.filter((m, i) => bodyValid[i]);
        const hits = raycaster.intersectObjects(targets, false);
        return hits.length ? hits[0].object.userData.bodyIndex : -1;
    }
    const onBodyDown = (ev) => {
        if (currentLayer !== "body") return;
        const j = _bodyPick(ev);
        if (j < 0) return;
        bodyDrag = j; orbit.enabled = false;
        selJoint = j;
        if (depthSl) { depthSl.value = String(bodyXYZ[j*3+2] || 0); depthVal.textContent = Number(bodyXYZ[j*3+2] || 0).toFixed(2); }
        // Fix a camera-facing plane through the joint for the whole drag, and
        // record the grab offset so the joint keeps its position under the
        // pointer instead of snapping its centre to the cursor.
        _dragPoint.set(bodyXYZ[j*3], bodyXYZ[j*3+1], bodyXYZ[j*3+2]);
        camera.getWorldDirection(_camDir);
        _grabPlane.setFromNormalAndCoplanarPoint(_camDir, _dragPoint.clone());
        _setNDC(ev); raycaster.setFromCamera(_dragNDC, camera);
        if (raycaster.ray.intersectPlane(_grabPlane, _grabHit)) _grabOffset.copy(_dragPoint).sub(_grabHit);
        else _grabOffset.set(0, 0, 0);
        ev.preventDefault(); ev.stopPropagation();
    };
    const onBodyMove = (ev) => {
        if (currentLayer !== "body") return;
        if (bodyDrag < 0) {
            const j = _bodyPick(ev);
            for (let i = 0; i < 18; i++) bodyMeshes[i].material = (i === j ? bodyHoverMat : bodyMat);
            hoverLabel.textContent = j >= 0
                ? `joint ${j} · ${bodyNames[j] || "?"}`
                : "drag a body joint · right-click resets it";
            return;
        }
        _setNDC(ev);
        raycaster.setFromCamera(_dragNDC, camera);
        // Intersect the FIXED grab plane; add the grab offset → exact pointer track.
        if (raycaster.ray.intersectPlane(_grabPlane, _grabHit)) {
            const nx = _grabHit.x + _grabOffset.x;
            const ny = _grabHit.y + _grabOffset.y;
            const nz = _grabHit.z + _grabOffset.z;
            if (rotateLimb && _applyLimbRotate(bodyDrag, nx, ny, nz)) {
                // subtree rotated in _applyLimbRotate
            } else {
                bodyXYZ[bodyDrag*3] = nx; bodyXYZ[bodyDrag*3+1] = ny; bodyXYZ[bodyDrag*3+2] = nz;
                const [ix, iy] = _worldToImg(nx, ny, nz);
                try { opts.setBodyJoints?.({ [bodyDrag]: [ix, iy] }); } catch (_) {}
            }
            _syncBodyMeshes();
        }
        ev.preventDefault(); ev.stopPropagation();
    };
    const onBodyUp = () => { if (bodyDrag >= 0) { bodyDrag = -1; orbit.enabled = true; } };
    const onBodyContext = (ev) => {
        if (currentLayer !== "body") return;
        const j = _bodyPick(ev);
        if (j < 0) return;
        ev.preventDefault();
        try { opts.clearBodyJoint?.(j); } catch (_) {}
        _seedBody();
    };
    renderer.domElement.addEventListener("mousedown", onBodyDown);
    renderer.domElement.addEventListener("mousemove", onBodyMove);
    window.addEventListener("mouseup", onBodyUp);
    renderer.domElement.addEventListener("contextmenu", onBodyContext);
    cleanups.push(() => {
        renderer.domElement.removeEventListener("mousedown", onBodyDown);
        renderer.domElement.removeEventListener("mousemove", onBodyMove);
        window.removeEventListener("mouseup", onBodyUp);
        renderer.domElement.removeEventListener("contextmenu", onBodyContext);
    });
    setLayer(currentLayer);   // apply initial visibility

    // ── State plumbing ───────────────────────────────────────────────
    const _eulerState = {
        yaw:   headState.yaw   || 0,
        pitch: headState.pitch || 0,
        roll:  headState.roll  || 0,
    };
    function _applyEuler(partial) {
        if (partial.yaw   !== undefined) _eulerState.yaw   = partial.yaw;
        if (partial.pitch !== undefined) _eulerState.pitch = partial.pitch;
        if (partial.roll  !== undefined) _eulerState.roll  = partial.roll;
        const DEG = Math.PI / 180;
        // YXZ order matches the python _rot_matrix convention (yaw-then-
        // pitch-then-roll applied to canonical points).
        headGroup.rotation.set(
            -_eulerState.pitch * DEG,
            -_eulerState.yaw   * DEG,
             _eulerState.roll  * DEG,
            "YXZ",
        );
    }
    _applyEuler({});

    function refresh() {
        if (destroyed) return;
        // Body layer: reseed the skeleton from the (possibly new) frame.
        if (currentLayer === "body") { _seedBody(); return; }
        // Re-read landmarks + head pose from the host node and update.
        let lms = null;
        try { lms = opts.getLandmarks?.(); } catch (_) {}
        if (Array.isArray(lms) && lms.length === 68) {
            const xyz = _landmarksToXYZ(lms);
            for (let i = 0; i < 68; i++) {
                meshes[i].position.set(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
            }
            // Rebuild edge positions.
            for (let e = 0; e < IBUG_EDGES.length; e++) {
                const [a, b] = IBUG_EDGES[e];
                edgePos[e * 6 + 0] = meshes[a].position.x;
                edgePos[e * 6 + 1] = meshes[a].position.y;
                edgePos[e * 6 + 2] = meshes[a].position.z;
                edgePos[e * 6 + 3] = meshes[b].position.x;
                edgePos[e * 6 + 4] = meshes[b].position.y;
                edgePos[e * 6 + 5] = meshes[b].position.z;
            }
            edgeGeom.attributes.position.needsUpdate = true;
        }
        let pose = null;
        try { pose = opts.getHeadPose?.(); } catch (_) {}
        if (pose) {
            _applyEuler({ yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll });
            yawCtl.slider.value   = String(pose.yaw   ?? 0);
            yawCtl.val.textContent = Number(pose.yaw   ?? 0).toFixed(1);
            pitchCtl.slider.value = String(pose.pitch ?? 0);
            pitchCtl.val.textContent = Number(pose.pitch ?? 0).toFixed(1);
            rollCtl.slider.value  = String(pose.roll  ?? 0);
            rollCtl.val.textContent = Number(pose.roll  ?? 0).toFixed(1);
            headGroup.position.set(pose.tx || 0, pose.ty || 0, pose.tz || 0);
        }
    }
    refresh();

    // ── Resize handling ──────────────────────────────────────────────
    // rAF-debounced + threshold-guarded to prevent the "ResizeObserver loop
    // completed with undelivered notifications" feedback storm — setSize on
    // the WebGL renderer can reflow sceneHost's parent, which re-fires the
    // observer with new dimensions.
    let _roPending = false;
    let _roLastW = 0, _roLastH = 0;
    const ro = new ResizeObserver(() => {
        if (destroyed || _roPending) return;
        _roPending = true;
        requestAnimationFrame(() => {
            _roPending = false;
            if (destroyed) return;
            const w = Math.max(160, sceneHost.clientWidth);
            const h = Math.max(120, sceneHost.clientHeight);
            if (Math.abs(w - _roLastW) < 2 && Math.abs(h - _roLastH) < 2) return;
            _roLastW = w; _roLastH = h;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });
    });
    ro.observe(sceneHost);
    cleanups.push(() => ro.disconnect());

    // ── Animation loop (rAF-driven; cancelled on destroy) ────────────
    let rafId = 0;
    const tick = () => {
        if (destroyed) return;
        rafId = requestAnimationFrame(tick);
        orbit.update();
        renderer.render(scene, camera);
    };
    tick();
    cleanups.push(() => cancelAnimationFrame(rafId));

    return { destroy, refresh };
}
