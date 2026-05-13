
# ComfyUI-WanAnimatePreprocessV2 — Node Reference

This pack registers **11 nodes**.  Each section lists the node's category, return types and every input widget exposed in ComfyUI.


## `DepthPoseCannyCombinedV2` — Depth + Pose + Canny Combined (V2)

> Self-contained ControlNet preprocessor producing depth, pose, canny, normal, layout-combined preview, AND a weighted blended map.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE` → `depth_map, pose_map, canny_map, normal_map, combined_map, blended_map`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `images` | IMAGE | `""`  | Input video frames (B,H,W,3) float32 [0,1]. |
| `width` | INT | `832` (min 64, max 4096) | Output canvas width. |
| `height` | INT | `480` (min 64, max 4096) | Output canvas height. |
| `enable_depth` | BOOLEAN | `True`  | Run the depth pass. Requires at least ONE depth source wired. |
| `enable_pose` | BOOLEAN | `True`  | Run the pose pass. |
| `enable_canny` | BOOLEAN | `True`  | Run the canny pass. |
| `canny_threshold1` | INT | `100` (min 0, max 500) | Canny lower hysteresis threshold. |
| `canny_threshold2` | INT | `200` (min 0, max 500) | Canny upper hysteresis threshold. |
| `canny_aperture` | combo[3, 5, 7] | `3`  | Sobel aperture for Canny (odd: 3/5/7). |
| `depth_colorize` | BOOLEAN | `False`  | If true, colorize grayscale depth with INFERNO colormap. Skipped when external_depth_map is already RGB. |
| `depth_invert` | BOOLEAN | `False`  | Invert depth (1 - depth). Use when source produces 'far = bright' but you want 'near = bright' (typical ControlNet expectation). |
| `pose_detection_threshold` | FLOAT | `0.05` (min 0.0, max 1.0, step 0.01) | YOLO confidence threshold (only used when posemodel is wired). |
| `pose_draw_threshold` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) | Per-keypoint score threshold for drawing the skeleton. |
| `combined_layout` | combo[horizontal_3, vertical_3, grid_2x2, depth_only, pose_only, canny_only] | `"horizontal_3"`  | Layout for the combined output. grid_2x2 = depth \| pose // canny \| original. |
| `depth_backend` | combo[auto, external, built_in_midas, damodel_v2, da3, depthcrafter, depth_pro] | `"auto"`  | Which depth backend to use. 'auto' tries: external_depth_map -> any wired loader -> built_in_midas. 'built_in_midas' makes the node fully self-contained (downloads MiDaS small via torch.hub on first use, ~80MB). |
| `enable_normal` | BOOLEAN | `True`  | Compute Sobel-from-depth NORMAL map. No model required (uses depth pass output). |
| `normal_strength` | FLOAT | `1.0` (min 0.1, max 10.0, step 0.1) | Scales the Sobel gradients before normalisation. Higher = stronger normal contrast. |
| `blend_mode` | combo[none, weighted_avg, screen, linear_dodge, max, multiply, overlay, channel_split] | `"weighted_avg"`  | How to combine depth+pose+canny+normal into blended_map. linear_dodge=additive (sharp), screen=highlight-safe, channel_split=Fun-Control (R=depth/G=canny/B=pose). |
| `depth_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of depth in blended_map. |
| `pose_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of pose in blended_map. |
| `canny_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of canny in blended_map. |
| `normal_weight` | FLOAT | `0.5` (min 0.0, max 4.0, step 0.05) | Weight of normal map in blended_map. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `external_depth_map` | IMAGE | `""`  | Pre-computed depth IMAGE batch from ANY upstream node. Highest priority. |
| `damodel_v2` | DAMODEL | `""`  | DepthAnything V2 model bundle from kijai/ComfyUI-DepthAnythingV2 (DownloadAndLoadDepthAnythingV2Model). Models: ComfyUI/models/depthanything/. |
| `da3_model` | DA3MODEL | `""`  | DepthAnything V3 config bundle from PozzettiAndrea/ComfyUI-DepthAnythingV3. Use the V3 pack's Inference node and feed its IMAGE output into external_depth_map. Models: ComfyUI/models/depthanything3/. |
| `depthcrafter_model` | DEPTHCRAFTER_MODEL | `""`  | DepthCrafter bundle from akatz-ai/ComfyUI-DepthCrafter-Nodes. Temporally consistent video depth. Models: ComfyUI/models/depthcrafter/. |
| `depth_pro_model` | DEPTH_PRO_MODEL | `""`  | Depth-Pro bundle from spacepxl/ComfyUI-Depth-Pro. Metric depth. Models: ComfyUI/models/depth/ml-depth-pro/. |
| `posemodel` | POSEMODEL | `""`  | From ONNX Detection Model Loader (V2) or animal-pose loader. Used if enable_pose=True AND no external_pose_map wired. |
| `external_pose_map` | IMAGE | `""`  | Pre-rendered pose map from any upstream node (e.g. Fannovel16/comfyui_controlnet_aux DWPose / OpenPose / AnimalPose). Highest priority for pose. |
| `depthcrafter_steps` | INT | `5` (min 1, max 100) | DepthCrafter only: diffusion inference steps. |
| `depthcrafter_guidance` | FLOAT | `1.0` (min 0.1, max 10.0, step 0.1) | DepthCrafter only: classifier-free guidance. |
| `depthcrafter_window` | INT | `110` (min 1, max 200) | DepthCrafter only: temporal window size. |
| `depthcrafter_overlap` | INT | `25` (min 0, max 100) | DepthCrafter only: window overlap. |


## `DrawViTPoseV2` — Draw ViT Pose (V2)

> Render the detected skeleton, face landmarks, iris pupils and gaze arrows onto a clean canvas at the target Wan 2.2 latent resolution. Outputs an IMAGE batch ready to drop into a Wan-Animate sampler.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `IMAGE` → `pose_images`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `pose_data` | POSEDATA | `""`  | From Pose and Face Detection (V2). |
| `width` | INT | `832` (min 64, max 2048) | Render canvas width (px). Match the sampler latent size. |
| `height` | INT | `480` (min 64, max 2048) | Render canvas height (px). Match the sampler latent size. |
| `retarget_padding` | INT | `16` (min 0, max 512) | Padding (px) added around the body bbox when retargeting. Larger = more headroom for big motions. |
| `body_stick_width` | INT | `-1` (min -1, max 20) | Body skeleton stick width in px. -1 = auto from canvas size. |
| `hand_stick_width` | INT | `-1` (min -1, max 20) | Hand skeleton stick width in px. -1 = auto. |
| `draw_head` | BOOLEAN | `True`  | Draw the head/face skeleton (eyes, nose, ears). |
| `pose_draw_threshold` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) | Per-keypoint score threshold for drawing. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `draw_iris` | BOOLEAN | `True`  | Draw iris/pupil markers from MediaPipe iris_data. |
| `draw_gaze` | BOOLEAN | `True`  | Draw gaze direction arrows from iris_data. |
| `iris_radius` | INT | `4` (min 1, max 20) | Pupil circle radius in pixels. |
| `gaze_arrow_len` | INT | `30` (min 4, max 200) | Length of gaze direction arrow in pixels. |
| `iris_min_confidence` | FLOAT | `0.05` (min 0.0, max 1.0, step 0.01) | Skip iris frames whose detection confidence is below this. |
| `iris_color` | combo[white, magenta, yellow, green] | `"white"`  | Color of the drawn pupil; magenta gives strongest sampler signal. |


## `OnnxDetectionModelLoaderV2` — ONNX Detection Model Loader (V2)

> Load ONNX ViTPose + YOLO detection models for Wan 2.2 Animate preprocessing. Place model files in `ComfyUI/models/detection/`. Outputs a `POSEMODEL` bundle that the detection node consumes.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `POSEMODEL` → `model`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vitpose_model` | combo[anim\vitpose-b-apt36k.onnx, anim\yolov8m.onnx, model.onnx, vitpose_h_wholebody_data.bin, vitpose_h_wholebody_model.onnx, yolov10m.onnx] | `""`  | ViTPose ONNX file (e.g. vitpose-h.onnx). Place in ComfyUI/models/detection/. |
| `yolo_model` | combo[anim\vitpose-b-apt36k.onnx, anim\yolov8m.onnx, model.onnx, vitpose_h_wholebody_data.bin, vitpose_h_wholebody_model.onnx, yolov10m.onnx] | `""`  | YOLO person-detector ONNX file. Place in ComfyUI/models/detection/. |
| `onnx_device` | combo[CUDAExecutionProvider, CPUExecutionProvider] | `"CUDAExecutionProvider"`  | Execution provider for ONNX Runtime. CUDA is much faster; CPU is the safe fallback. |


## `PoseAndFaceDetectionV2` — Pose and Face Detection (V2)

> Run YOLO person detection + ViTPose 2D keypoints + (optional) MediaPipe FaceMesh on a video tensor. Produces the full pose/face/iris bundle required by Wan 2.2 Animate Character Replacement workflows.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `POSEDATA, IMAGE, STRING, BBOX, BBOX, STRING, IMAGE, STRING, STRING, FLOAT, FACE_RESTORE_INFO, FLOAT` → `pose_data, face_images, key_frame_body_points, bboxes, face_bboxes, iris_data, debug_image, right_pupil_xy, left_pupil_xy, lip_openness_ratio, restore_info, face_cfg_scale`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `model` | POSEMODEL | `""`  | From ONNX Detection Model Loader (V2). |
| `images` | IMAGE | `""`  | Video frames as an IMAGE batch (B,H,W,C float [0,1]). |
| `width` | INT | `832` (min 64, max 2048) | Target canvas width (px) used for retarget math. Match your Wan 2.2 latent size. |
| `height` | INT | `480` (min 64, max 2048) | Target canvas height (px). Match your Wan 2.2 latent size. |
| `detection_threshold` | FLOAT | `0.05` (min 0.0, max 1.0, step 0.01) | YOLO confidence threshold. Lower = more permissive person detection. |
| `pose_threshold` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) | Per-keypoint score threshold. Below this a keypoint is treated as missing. |
| `use_clahe` | BOOLEAN | `True`  | Apply CLAHE contrast enhancement for pose detection. |
| `use_blur_for_pose` | BOOLEAN | `True`  | Apply Gaussian blur internally for YOLO and ViTPose. |
| `blur_radius` | INT | `5` (min 1, max 20, step 1) | Gaussian blur kernel radius applied to the face mask edge to soften the boundary. Higher = wider feather. Kernel size = radius*2+1 px. |
| `blur_sigma` | FLOAT | `2.0` (min 0.1, max 5.0, step 0.1) | Gaussian blur sigma (standard deviation) for the face mask feather. Higher sigma = softer falloff. Tune together with blur_radius. |
| `use_face_smoothing` | BOOLEAN | `True`  | Smooth face bounding box center over time. |
| `face_smoothing_strength` | FLOAT | `0.6` (min 0.0, max 1.0, step 0.05) | Higher = more smoothing |
| `use_constant_face_box` | BOOLEAN | `True`  | Keep a constant pixel size face crop; position adapts. |
| `face_box_size_px` | INT | `224` (min 64, max 1024, step 16) | Pixel size of the square face crop when constant mode is on. |
| `use_iris_smoothing` | BOOLEAN | `True`  | Temporally smooth iris pixel positions across frames. Reduces per-frame jitter that Wan 2.2 Animate's face encoder picks up and reproduces as wobbly gaze. |
| `iris_smoothing_strength` | FLOAT | `0.4` (min 0.0, max 1.0, step 0.05) | EMA mix weight when iris_smoothing_method='ema'. Higher = more smoothing, more lag. Ignored for one_euro / none. |
| `iris_smoothing_method` | combo[one_euro, ema, none] | `"one_euro"`  | Iris pixel-position smoother. one_euro = adaptive low-pass (Casiez 2012, recommended). ema = legacy first-order; tweak via iris_smoothing_strength. none = raw per-frame positions. |
| `iris_one_euro_min_cutoff` | FLOAT | `1.0` (min 0.05, max 10.0, step 0.05) | One-euro min cutoff (Hz) for iris pixel coords. Lower = stronger jitter rejection on near-static eyes (small saccades preserved). |
| `iris_one_euro_beta` | FLOAT | `0.05` (min 0.0, max 5.0, step 0.01) | One-euro speed coefficient for iris pixel coords. Higher = filter relaxes faster on quick eye movements; lower = stronger steady-state smoothing. |
| `gaze_lock_eyes` | BOOLEAN | `True`  | Couple left & right eye gaze so they always look in the SAME direction. Both eyes' yaw/pitch are blended toward their per-frame average. Single most effective fix for the 'eyes pointing different directions' artefact in Wan 2.2 Animate output. |
| `gaze_lock_strength` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.05) | How strongly to pull each eye toward the shared average. 0 = independent (legacy). 1 = perfectly conjugate (both eyes always parallel). 0.7 keeps a touch of natural convergence/divergence. |
| `use_mediapipe_face` | BOOLEAN | `True`  | Use MediaPipe FaceMesh (478 pts incl. iris/lips) to override face landmarks. Falls back to ViTPose pupil voting if MediaPipe is missing or fails on a frame. |
| `use_blendshape_gaze` | BOOLEAN | `True`  | Use MediaPipe FaceLandmarker (Tasks API) blend shapes for production-grade per-eye yaw/pitch in radians. Head-pose-corrected by training. Auto-downloads face_landmarker.task (~3MB) on first run. Falls back to legacy 2D iris-offset gaze if disabled or unavailable. |
| `gaze_one_euro_min_cutoff` | FLOAT | `1.7` (min 0.05, max 10.0, step 0.05) | One-euro filter base cutoff frequency (Hz). Lower = more aggressive jitter rejection at the cost of slight lag. 1.7 is a good default for 24-30 fps gaze. |
| `gaze_one_euro_beta` | FLOAT | `0.3` (min 0.0, max 5.0, step 0.05) | One-euro filter speed coefficient. Higher = filter relaxes faster on quick saccades, preserving responsiveness; lower = stronger smoothing during fast moves. |
| `gaze_max_yaw_deg` | FLOAT | `30.0` (min 5.0, max 60.0, step 1.0) | Saturation yaw angle in degrees that corresponds to blend shape value 1.0. 30° covers the comfortable physiological range; raise for more dramatic eye motion. |
| `gaze_max_pitch_deg` | FLOAT | `25.0` (min 5.0, max 60.0, step 1.0) | Saturation pitch angle in degrees that corresponds to blend shape value 1.0. 25° covers the comfortable physiological range. |
| `crop_mode` | combo[default, auto, jitterless] | `"default"`  | default = raw detected bbox per frame (NO smoothing / NO constant size — crop is effectively 'off'). auto = legacy smoothed + optional constant-size box. jitterless = lock crop SIZE from frame 0, smoothly track the CENTER, allow manual frame-0 + key-frame overrides. |
| `frame0_cx` | INT | `-1` (min -1, max 8192) | Frame 0 anchor center X in pixels. -1 = use detected face center on frame 0. Used only when crop_mode=jitterless. |
| `frame0_cy` | INT | `-1` (min -1, max 8192) | Frame 0 anchor center Y in pixels. -1 = use detected face center on frame 0. |
| `frame0_size` | INT | `0` (min 0, max 4096, step 16) | Locked square crop size in pixels (used for the entire clip). 0 = fall back to face_box_size_px. |
| `keyframes_json` | STRING | `"[]"`  | JSON list of per-frame overrides: [{"frame":N, "cx":X, "cy":Y, "size":S?}, ...]. Frames between key-frames are linearly interpolated. size is optional; if omitted the locked size is kept. |
| `smoothing_method` | combo[one_euro, ema, gaussian, none] | `"one_euro"`  | Center-trajectory filter. one_euro = jitterless adaptive low-pass (recommended). ema = legacy motion-adaptive EMA. gaussian = fixed-window 1D blur. none = raw. |
| `crop_one_euro_min_cutoff` | FLOAT | `1.0` (min 0.05, max 10.0, step 0.05) | One-euro min cutoff (Hz) for crop center. Lower = stronger jitter rejection. |
| `crop_one_euro_beta` | FLOAT | `0.05` (min 0.0, max 5.0, step 0.01) | One-euro speed coefficient for crop center. Higher = filter relaxes faster on quick motion. |
| `crop_gaussian_window` | INT | `7` (min 3, max 51, step 2) | Window size (odd) for the Gaussian temporal blur of the crop center. |
| `eye_align_mode` | combo[default, eye_upper_third] | `"default"`  | Wan-Animate paper recommendation #1: 'eye_upper_third' vertically shifts the face crop so eyes land at the upper third of the 512x512 face encoder input. The encoder reads holistic face appearance, so consistent eye placement directly improves gaze fidelity. 'default' keeps legacy bbox center. |
| `eye_y_fraction` | FLOAT | `0.3` (min 0.1, max 0.6, step 0.01) | Target eye row as a fraction of crop height (0.30 = upper third). Only used when eye_align_mode = 'eye_upper_third'. |
| `face_cfg_scale` | FLOAT | `1.0` (min 1.0, max 10.0, step 0.1) | Wan-Animate paper recommendation #3 (paper section 4.3): CFG on the face conditioning input gives finer control over expression / gaze when finer reenactment is desired. This widget is a passthrough -- wire the FLOAT output 'face_cfg_scale' into your Wan-Animate sampler's face CFG input. 1.0 = CFG disabled (default, fastest). 2.0-4.0 = stronger expression adherence. >5.0 may over-saturate. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `bbox_override` | BBOX | `""`  | Optional external BBOX for the frame-0 anchor. Highest priority; overrides frame0_cx/cy/size widgets. |


## `SelfContainedControlNetPreprocessorV2` — Self-Contained ControlNet Preprocessor (V2)

> Self-contained ControlNet preprocessor producing depth, pose, canny, normal, layout-combined preview, AND a weighted blended map.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE` → `depth_map, pose_map, canny_map, normal_map, combined_map, blended_map`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `images` | IMAGE | `""`  | Input video frames (B,H,W,3) float32 [0,1]. |
| `width` | INT | `832` (min 64, max 4096) | Output canvas width. |
| `height` | INT | `480` (min 64, max 4096) | Output canvas height. |
| `enable_depth` | BOOLEAN | `True`  | Run the depth pass. Requires at least ONE depth source wired. |
| `enable_pose` | BOOLEAN | `True`  | Run the pose pass. |
| `enable_canny` | BOOLEAN | `True`  | Run the canny pass. |
| `canny_threshold1` | INT | `100` (min 0, max 500) | Canny lower hysteresis threshold. |
| `canny_threshold2` | INT | `200` (min 0, max 500) | Canny upper hysteresis threshold. |
| `canny_aperture` | combo[3, 5, 7] | `3`  | Sobel aperture for Canny (odd: 3/5/7). |
| `depth_colorize` | BOOLEAN | `False`  | If true, colorize grayscale depth with INFERNO colormap. Skipped when external_depth_map is already RGB. |
| `depth_invert` | BOOLEAN | `False`  | Invert depth (1 - depth). Use when source produces 'far = bright' but you want 'near = bright' (typical ControlNet expectation). |
| `pose_detection_threshold` | FLOAT | `0.05` (min 0.0, max 1.0, step 0.01) | YOLO confidence threshold (only used when posemodel is wired). |
| `pose_draw_threshold` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) | Per-keypoint score threshold for drawing the skeleton. |
| `combined_layout` | combo[horizontal_3, vertical_3, grid_2x2, depth_only, pose_only, canny_only] | `"horizontal_3"`  | Layout for the combined output. grid_2x2 = depth \| pose // canny \| original. |
| `depth_backend` | combo[auto, external, built_in_midas, damodel_v2, da3, depthcrafter, depth_pro] | `"auto"`  | Which depth backend to use. 'auto' tries: external_depth_map -> any wired loader -> built_in_midas. 'built_in_midas' makes the node fully self-contained (downloads MiDaS small via torch.hub on first use, ~80MB). |
| `enable_normal` | BOOLEAN | `True`  | Compute Sobel-from-depth NORMAL map. No model required (uses depth pass output). |
| `normal_strength` | FLOAT | `1.0` (min 0.1, max 10.0, step 0.1) | Scales the Sobel gradients before normalisation. Higher = stronger normal contrast. |
| `blend_mode` | combo[none, weighted_avg, screen, linear_dodge, max, multiply, overlay, channel_split] | `"weighted_avg"`  | How to combine depth+pose+canny+normal into blended_map. linear_dodge=additive (sharp), screen=highlight-safe, channel_split=Fun-Control (R=depth/G=canny/B=pose). |
| `depth_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of depth in blended_map. |
| `pose_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of pose in blended_map. |
| `canny_weight` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Weight of canny in blended_map. |
| `normal_weight` | FLOAT | `0.5` (min 0.0, max 4.0, step 0.05) | Weight of normal map in blended_map. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `external_depth_map` | IMAGE | `""`  | Pre-computed depth IMAGE batch from ANY upstream node. Highest priority. |
| `damodel_v2` | DAMODEL | `""`  | DepthAnything V2 model bundle from kijai/ComfyUI-DepthAnythingV2 (DownloadAndLoadDepthAnythingV2Model). Models: ComfyUI/models/depthanything/. |
| `da3_model` | DA3MODEL | `""`  | DepthAnything V3 config bundle from PozzettiAndrea/ComfyUI-DepthAnythingV3. Use the V3 pack's Inference node and feed its IMAGE output into external_depth_map. Models: ComfyUI/models/depthanything3/. |
| `depthcrafter_model` | DEPTHCRAFTER_MODEL | `""`  | DepthCrafter bundle from akatz-ai/ComfyUI-DepthCrafter-Nodes. Temporally consistent video depth. Models: ComfyUI/models/depthcrafter/. |
| `depth_pro_model` | DEPTH_PRO_MODEL | `""`  | Depth-Pro bundle from spacepxl/ComfyUI-Depth-Pro. Metric depth. Models: ComfyUI/models/depth/ml-depth-pro/. |
| `posemodel` | POSEMODEL | `""`  | From ONNX Detection Model Loader (V2) or animal-pose loader. Used if enable_pose=True AND no external_pose_map wired. |
| `external_pose_map` | IMAGE | `""`  | Pre-rendered pose map from any upstream node (e.g. Fannovel16/comfyui_controlnet_aux DWPose / OpenPose / AnimalPose). Highest priority for pose. |
| `depthcrafter_steps` | INT | `5` (min 1, max 100) | DepthCrafter only: diffusion inference steps. |
| `depthcrafter_guidance` | FLOAT | `1.0` (min 0.1, max 10.0, step 0.1) | DepthCrafter only: classifier-free guidance. |
| `depthcrafter_window` | INT | `110` (min 1, max 200) | DepthCrafter only: temporal window size. |
| `depthcrafter_overlap` | INT | `25` (min 0, max 100) | DepthCrafter only: window overlap. |


## `WanAnimateFaceQualityCheckV2` — Wan-Animate Face Quality Check (V2)

> Score each face crop on (a) Laplacian-variance sharpness and (b) eye-region brightness, then optionally repair bad frames by copying the previous good frame or by simple sharpening. Bad face conditioning frames cause the Wan-Animate face encoder to produce drifting / wrong-direction gaze (paper Sec. 4.3). Connect this BETWEEN Pose-and-Face-Detection (V2)'s `face_images` output and your downstream face-id encoder.

- **Category:** `WanAnimatePreprocess_V2`
- **Returns:** `IMAGE, FLOAT, STRING` → `face_images_repaired, good_frame_ratio, report_json`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `face_images` | IMAGE | `""`  | Per-frame 512x512 face crops (output of Pose and Face Detection V2). |
| `blur_threshold` | FLOAT | `50.0` (min 0.0, max 5000.0, step 1.0) | Laplacian-variance threshold below which a frame is flagged as blurry. Typical sharp 512x512 frames score 100-1000; <50 indicates motion blur or out-of-focus. |
| `min_eye_brightness` | FLOAT | `0.1` (min 0.0, max 1.0, step 0.01) | Minimum mean luma of the eye-region strip (rows 30%-55%). Below this, eyes are likely closed or the frame is too dark for the encoder to read gaze. |
| `auto_repair_bad_frames` | BOOLEAN | `True`  | If true, repair frames flagged as bad. If false, just report stats. |
| `repair_strategy` | combo[copy_previous_good, unsharp_mask, skip] | `"copy_previous_good"`  | copy_previous_good: replace with last good frame. unsharp_mask: deconvolve-style sharpening. skip: leave untouched but report. |


## `WanExpressionCoefficientsV2` — Wan Expression Coefficients (V2)

> Extract 52 ARKit expression coefficients + temporal smoothing + stability scoring from iris_data JSON.

- **Category:** `WanAnimatePreprocess_V2/Expression`
- **Returns:** `STRING, STRING, FLOAT, INT` → `coeffs_json, smoothed_json, mean_stability, active_count`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `iris_data_json` | STRING | `"[]"`  |  |
| `fps` | FLOAT | `30.0` (min 1.0, max 240.0, step 1.0) |  |
| `smooth_min_cutoff` | FLOAT | `1.5` (min 0.01, max 30.0, step 0.1) |  |
| `smooth_beta` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.001) |  |
| `active_threshold` | FLOAT | `0.1` (min 0.0, max 1.0, step 0.01) |  |
| `topk` | INT | `0` (min 0, max 52) | 0 = keep all; >0 = keep top-K by variance. |


## `WanHeadPose6DoFV2` — Wan Head Pose 6DoF — solvePnP (V2)

> 6-DoF head pose via cv2.solvePnP on MediaPipe 478 landmarks (14-point canonical model).

- **Category:** `WanAnimatePreprocess_V2/HeadPose`
- **Returns:** `STRING, IMAGE, FLOAT, FLOAT, FLOAT` → `poses_json, overlay, yaw, pitch, roll`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `landmarks_json` | STRING | `"[]"`  | JSON: list of frames; each frame is a list of [x,y] (length 478) or a dict with face_landmarks/kps_face. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  |  |
| `image_width_override` | INT | `0` (min 0, max 16384) |  |
| `image_height_override` | INT | `0` (min 0, max 16384) |  |
| `axis_length` | FLOAT | `6.0` (min 1.0, max 30.0, step 0.5) |  |


## `WanIrisControlNetV2` — Wan Iris ControlNet Conditioning (V2)

> Render an iris/gaze ControlNet conditioning image from iris_data JSON. Eye masks, iris discs, gaze arrows, gaze-target heatmap.

- **Category:** `WanAnimatePreprocess_V2/Gaze`
- **Returns:** `IMAGE, MASK, STRING` → `control_image, iris_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `iris_data_json` | STRING | `"[]"`  |  |
| `image_width` | INT | `1024` (min 64, max 8192) |  |
| `image_height` | INT | `1024` (min 64, max 8192) |  |
| `render_style` | combo[full, iris_only, heatmap_only, mask_only] | `"full"`  |  |
| `iris_radius_px` | INT | `6` (min 1, max 80) |  |
| `arrow_scale_px` | FLOAT | `80.0` (min 0.0, max 400.0, step 5.0) | Pixels of arrow per radian of gaze. |
| `heatmap_sigma_px` | FLOAT | `35.0` (min 1.0, max 400.0) |  |
| `background` | combo[black, white, neutral_grey] | `"black"`  |  |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `face_bboxes` | BBOX | `""`  |  |
| `reference_image` | IMAGE | `""`  | If given, use its (H,W,B) and overlay onto it at low alpha. |
| `overlay_alpha` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.05) |  |


## `WanQualityScorerJitterV2` — Wan Quality Scorer — Temporal Jitter (V2)

> Per-frame quality and jitter metrics from pose + (optional) expression JSON. Outputs aggregate quality score in [0,1].

- **Category:** `WanAnimatePreprocess_V2/Quality`
- **Returns:** `STRING, FLOAT, FLOAT, FLOAT, FLOAT, INT` → `metrics_json, quality_score, mean_body_velocity, mean_face_velocity, mean_expression_jitter, bad_frame_count`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `pose_data_json` | STRING | `"[]"`  |  |
| `image_diagonal_px` | FLOAT | `1500.0` (min 64.0, max 16384.0, step 1.0) | Used to normalise pixel velocities. |
| `confidence_threshold` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) |  |
| `max_velocity_px` | FLOAT | `60.0` (min 1.0, max 1000.0, step 1.0) | Body kp px velocity that would yield score 0. |
| `expression_window` | INT | `8` (min 2, max 60) |  |
| `bad_velocity_thr_px` | FLOAT | `40.0` (min 1.0, max 1000.0, step 1.0) |  |
| `bad_visibility_thr` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.05) |  |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `expression_coeffs_json` | STRING | `""`  |  |


## `WanSHLightingTransferV2` — Wan SH Lighting Transfer (V2)

> Spherical-harmonics lighting fit (L=2, 9 basis, per RGB channel) with optional relighting onto a target. Basri-Jacobs/Ramamoorthi formulation.

- **Category:** `WanAnimatePreprocess_V2/Lighting`
- **Returns:** `IMAGE, STRING, IMAGE` → `relit_image, sh_coeffs_json, shading_map`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `source_image` | IMAGE | `""`  |  |
| `source_normal` | IMAGE | `""`  |  |
| `operation` | combo[fit_only, transfer, rotate_lights] | `"fit_only"`  |  |
| `rotate_yaw_deg` | FLOAT | `0.0` (min -180.0, max 180.0, step 1.0) |  |
| `intensity` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) |  |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `source_albedo` | IMAGE | `""`  |  |
| `target_image` | IMAGE | `""`  |  |
| `target_normal` | IMAGE | `""`  |  |
| `target_albedo` | IMAGE | `""`  |  |
