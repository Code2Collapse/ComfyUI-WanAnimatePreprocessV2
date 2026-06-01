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

// CDN base for Three.js. Pinned version for reproducibility.
// unpkg serves the ESM bundle + jsm examples with correct MIME so dynamic
// import() works directly without a bundler.
const THREE_VERSION = "0.158.0";
const CDN_BASES = [
    `https://unpkg.com/three@${THREE_VERSION}`,
    `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}`,
];

async function loadThreeStack() {
    let lastErr = null;
    for (const base of CDN_BASES) {
        try {
            const THREE = await import(/* @vite-ignore */ `${base}/build/three.module.js`);
            // Import-map shim: OrbitControls/TransformControls import "three"
            // by bare name. We resolve that to the same CDN URL by passing
            // the THREE module directly into a wrapper that side-loads them
            // as raw text + new Function. Simpler: fetch as text and exec
            // with `three` already in scope.
            const fetchModule = async (path) => {
                const url = `${base}/examples/jsm/${path}`;
                const txt = await fetch(url).then((r) => {
                    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
                    return r.text();
                });
                // Rewrite the bare `from "three"` import to the explicit CDN URL
                // so the resulting module resolves cleanly.
                const patched = txt.replace(
                    /from\s*['"]three['"]/g,
                    `from '${base}/build/three.module.js'`,
                );
                const blob = new Blob([patched], { type: "text/javascript" });
                const blobUrl = URL.createObjectURL(blob);
                try {
                    return await import(/* @vite-ignore */ blobUrl);
                } finally {
                    // Free the blob after import resolution (the module is
                    // cached by the browser by URL; revoking is safe).
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
            // Try the next CDN.
        }
    }
    const err = new Error("Three.js CDN load failed: " + (lastErr ? lastErr.message : "unknown"));
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
    const wrap = document.createElement("div");
    wrap.style.cssText =
        "position:relative;width:100%;height:340px;display:flex;flex-direction:column;" +
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
    subtitle.textContent = "loading Three.js\u2026";
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u2715";
    closeBtn.title = "Close 3D editor";
    closeBtn.style.cssText =
        "margin-left:auto;background:transparent;color:" + C.text + ";" +
        "border:1px solid " + C.border + ";border-radius:3px;" +
        "padding:1px 6px;cursor:pointer;";
    hdr.append(title, subtitle, closeBtn);
    wrap.appendChild(hdr);

    // Loading placeholder (shown until Three.js resolves or fails).
    const loadingMsg = document.createElement("div");
    loadingMsg.style.cssText =
        "flex:1;display:flex;align-items:center;justify-content:center;color:" + C.dim + ";";
    loadingMsg.textContent = "fetching three@" + THREE_VERSION + " from CDN\u2026";
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
    subtitle.textContent = "three@" + THREE_VERSION + " \u00b7 " + cdnBase.replace(/^https?:\/\//, "");
    loadingMsg.remove();

    // ── Three.js scene setup ─────────────────────────────────────────
    const sceneHost = document.createElement("div");
    sceneHost.style.cssText = "flex:1;position:relative;min-height:0;";
    wrap.appendChild(sceneHost);

    const W0 = Math.max(160, sceneHost.clientWidth  || wrap.clientWidth || 320);
    const H0 = Math.max(120, sceneHost.clientHeight || 240);

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
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(1, 1, 2);
    scene.add(dir);

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
    const ro = new ResizeObserver(() => {
        if (destroyed) return;
        const w = Math.max(160, sceneHost.clientWidth);
        const h = Math.max(120, sceneHost.clientHeight);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
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
