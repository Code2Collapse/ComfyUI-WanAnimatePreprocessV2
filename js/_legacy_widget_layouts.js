/**
 * Saved-workflow layout repair for nodes whose inputs were inserted mid-list (agy review 2026-11-28, P0).
 *
 * ComfyUI saves widget values BY POSITION. PoseAndFaceDetectionV2 gained clahe_clip_limit ... detect_saturation in
 * the middle of its inputs (2026-07-29) and DrawViTPoseV2's face_cfg_scale became connection-only, so an older save
 * put every later value into the wrong widget (example_workflows/Pose.extraction.json: use_blur_for_pose=True landed
 * on clahe_clip_limit, face crop 224 on detect_saturation).
 *
 * LEGACY below is every earlier widget layout, generated from this repo's git history of nodes.py (since = first
 * commit with that layout). On load the saved array is matched to the layout it fits best (length, then value
 * types); when that is an old layout, values go back BY NAME and widgets the old layout did not have return to their
 * defaults. Saves in the current layout are left alone.
 */
const app = window.comfyAPI?.app?.app;

// generated from git history (agy WAPV2 review P0): {node: [{since, order}]}, newest first
export const LEGACY = {"PoseAndFaceDetectionV2": [{"since": "86c8d08", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "clahe_clip_limit", "clahe_grid_size", "detect_gamma", "detect_white_balance", "detect_denoise", "detect_sharpen", "detect_saturation", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_crop_scale", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "preserve_face_aspect", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json", "face_resize_filter", "face_sr", "face_sr_model", "face_sr_stabilise", "use_flux", "flux_kontext_path"]}, {"since": "c99e637", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "clahe_clip_limit", "clahe_grid_size", "detect_gamma", "detect_white_balance", "detect_denoise", "detect_sharpen", "detect_saturation", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_crop_scale", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "preserve_face_aspect", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json", "face_sr", "face_sr_model", "face_sr_stabilise", "use_flux", "flux_kontext_path"]}, {"since": "d219761", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "clahe_clip_limit", "clahe_grid_size", "detect_gamma", "detect_white_balance", "detect_denoise", "detect_sharpen", "detect_saturation", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_crop_scale", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "preserve_face_aspect", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json", "use_flux", "flux_kontext_path"]}, {"since": "18e0998", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "clahe_clip_limit", "clahe_grid_size", "detect_gamma", "detect_white_balance", "detect_denoise", "detect_sharpen", "detect_saturation", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_crop_scale", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "preserve_face_aspect", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json"]}, {"since": "f98273e", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "clahe_clip_limit", "clahe_grid_size", "detect_gamma", "detect_white_balance", "detect_denoise", "detect_sharpen", "detect_saturation", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "preserve_face_aspect", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json"]}, {"since": "1d9aec3", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "crop_safety_margin", "crop_size_one_euro_beta", "crop_containment_check", "crop_containment_tolerance", "auto_smoothing_method", "force_eyes_open", "eye_open_mode", "eye_open_blink_ear", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json"]}, {"since": "cb5b8ab", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "export_expression_coeffs", "landmark_overrides_json"]}, {"since": "d21e679", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "au_amplify", "au_amplify_neutral_frame", "landmark_overrides_json"]}, {"since": "0cc59af", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image", "landmark_overrides_json"]}, {"since": "d097cf7", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "gaze_calibration_frame", "apply_gaze_to_face_image"]}, {"since": "c5db68d", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps", "apply_gaze_to_face_image"]}, {"since": "644e6a9", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale", "gaze_engine", "gaze_kalman_meas_std_deg", "gaze_kalman_process_std", "gaze_fps"]}, {"since": "cb661aa", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "iris_smoothing_method", "iris_one_euro_min_cutoff", "iris_one_euro_beta", "gaze_lock_eyes", "gaze_lock_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window", "eye_align_mode", "eye_y_fraction", "face_cfg_scale"]}, {"since": "86243da", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg", "crop_mode", "frame0_cx", "frame0_cy", "frame0_size", "keyframes_json", "smoothing_method", "crop_one_euro_min_cutoff", "crop_one_euro_beta", "crop_gaussian_window"]}, {"since": "367437e", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "use_mediapipe_face", "use_blendshape_gaze", "gaze_one_euro_min_cutoff", "gaze_one_euro_beta", "gaze_max_yaw_deg", "gaze_max_pitch_deg"]}, {"since": "6bc8e97", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength", "use_mediapipe_face"]}, {"since": "985ca88", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px", "use_iris_smoothing", "iris_smoothing_strength"]}, {"since": "a9c6f8d", "order": ["width", "height", "detection_threshold", "pose_threshold", "use_clahe", "use_blur_for_pose", "blur_radius", "blur_sigma", "use_face_smoothing", "face_smoothing_strength", "use_constant_face_box", "face_box_size_px"]}, {"since": "upstream", "order": ["width", "height"]}], "DrawViTPoseV2": [{"since": "e9fa28f", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head", "pose_draw_threshold", "draw_iris", "draw_gaze", "iris_radius", "gaze_arrow_len", "iris_min_confidence", "iris_color", "face_cfg_scale", "enforce_512_face", "reference_expression_coeffs_json", "segment_length", "top_k_aus", "apply_pose_edits_to_face"]}, {"since": "50bd654", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head", "pose_draw_threshold", "draw_iris", "draw_gaze", "iris_radius", "gaze_arrow_len", "iris_min_confidence", "iris_color", "face_cfg_scale", "enforce_512_face", "reference_expression_coeffs_json", "segment_length", "top_k_aus"]}, {"since": "d21e679", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head", "pose_draw_threshold", "draw_iris", "draw_gaze", "iris_radius", "gaze_arrow_len", "iris_min_confidence", "iris_color", "face_cfg_scale", "enforce_512_face"]}, {"since": "4da4317", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head", "pose_draw_threshold", "draw_iris", "draw_gaze", "iris_radius", "gaze_arrow_len", "iris_min_confidence", "iris_color"]}, {"since": "9417da2", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head", "pose_draw_threshold"]}, {"since": "upstream", "order": ["width", "height", "retarget_padding", "body_stick_width", "hand_stick_width", "draw_head"]}]};

function kindOf(spec) {
    if (!Array.isArray(spec)) return null;
    const t = spec[0];
    if (t === "INT" || t === "FLOAT") return "num";
    if (t === "BOOLEAN") return "bool";
    if (t === "STRING" || t === "COMBO" || Array.isArray(t)) return "str";
    return null;
}

function fits(kind, v) {
    if (v === null || v === undefined) return false;
    if (kind === "num") return typeof v === "number";
    if (kind === "bool") return typeof v === "boolean" || v === "True" || v === "False";
    if (kind === "str") return typeof v === "string";
    return true;
}

function specsOf(nodeData) {
    return { ...(nodeData?.input?.required || {}), ...(nodeData?.input?.optional || {}) };
}

export function currentOrder(nodeData) {
    const order = [];
    for (const sect of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(nodeData?.input?.[sect] || {})) {
            if (kindOf(spec) && !(spec[1] && spec[1].forceInput)) order.push(name);
        }
    }
    return order;
}

function score(order, vals, specs) {
    let s = 0;
    order.forEach((name, i) => {
        const k = kindOf(specs[name]);
        if (k) s += fits(k, vals[i]) ? 1 : -2;      // an input that no longer exists scores 0
    });
    return s;
}

/** The layout a saved array was written with, or null when the current layout fits at least as well. */
export function pickLegacyLayout(nodeName, nodeData, saved) {
    const layouts = LEGACY[nodeName];
    if (!layouts || !Array.isArray(saved)) return null;
    const specs = specsOf(nodeData);
    const cur = currentOrder(nodeData);
    // a display widget saved before it was marked serialize=false left one trailing value (pose_gaze_view)
    const variants = [saved];
    const last = saved[saved.length - 1];
    if (saved.length && (last === null || last === undefined || typeof last === "object")) variants.push(saved.slice(0, -1));
    let best = null;
    for (const vals of variants) {
        if (vals.length === cur.length) {
            const s = score(cur, vals, specs);
            if (!best || s >= best.score) best = { score: s, order: null, vals };
        }
        for (const l of layouts) {
            if (l.order.length !== vals.length) continue;
            const s = score(l.order, vals, specs);
            if (!best || s > best.score) best = { score: s, order: l.order, since: l.since, vals };
        }
    }
    return best && best.order ? best : null;
}

function coerce(kind, v) {
    if (kind === "bool" && (v === "True" || v === "False")) return v === "True";
    return v;
}

/** Put the saved values back by name; widgets the old layout lacked get their defaults. Returns changed names. */
export function applyLegacyLayout(node, nodeData, pick) {
    const specs = specsOf(nodeData);
    const saved = new Map(pick.order.map((n, i) => [n, pick.vals[i]]));
    const moved = [];
    for (const name of currentOrder(nodeData)) {
        const w = node.widgets?.find((x) => x.name === name);
        if (!w) continue;
        const spec = specs[name];
        const k = kindOf(spec);
        let v;
        if (saved.has(name) && fits(k, saved.get(name))) v = coerce(k, saved.get(name));
        else {
            const d = spec?.[1]?.default;
            v = d !== undefined ? d : (Array.isArray(spec?.[0]) ? spec[0][0] : w.value);
        }
        if (w.value !== v) { w.value = v; moved.push(name); }
    }
    return moved;
}

app?.registerExtension({
    name: "WAPV2.LegacyWidgetLayouts",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LEGACY[nodeData?.name]) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            // stash the array as saved: other extensions may pad or rewrite info.widgets_values during configure
            try { this.__wapSavedWV = Array.isArray(info?.widgets_values) ? info.widgets_values.slice() : null; }
            catch (_e) { this.__wapSavedWV = null; }
            return configure.apply(this, arguments);
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const saved = this.__wapSavedWV;
            this.__wapSavedWV = null;
            const pick = pickLegacyLayout(nodeData.name, nodeData, saved);
            if (pick) {
                // after the frontend's positional pass and any same-tick fix-ups by other extensions
                setTimeout(() => {
                    try {
                        const moved = applyLegacyLayout(this, nodeData, pick);
                        console.warn(`[WAPV2] ${nodeData.name} #${this.id}: saved with an older input layout `
                            + `(${pick.since}); restored ${moved.length} value(s) by name. Save the workflow to keep it.`);
                        this.setDirtyCanvas?.(true, true);
                    } catch (e) { console.warn("[WAPV2] legacy layout repair failed:", e); }
                }, 0);
            }
            return r;
        };
    },
});
