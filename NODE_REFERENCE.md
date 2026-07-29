# Wan-Animate Preprocess V2 — Node Reference

*Auto-generated from the live `NODE_CLASS_MAPPINGS` on 2026-07-29 — 16 nodes. Every parameter description below is the node's own tooltip, so this file cannot drift from the code.*

Regenerate after changing any node's `INPUT_TYPES`.


## Contents

- **WanAnimatePreprocess/Pose** (1)
  - [Wan Pose 3D Refine — NLF (V2)](#wanpose3drefinenlfv2)
- **WanAnimatePreprocessV2/extras** (4)
  - [Wan Face Controller 3D](#wanfacecontroller3dv2)
  - [Wan Gaze — ETH-XGaze Post-Processor (V2)](#wangazeethxgazev2)
  - [Wan Pose Detect — YOLO + ViTPose (V2)](#wanposedetectvitposev2)
  - [Wan Pose Format Convert — OP18 → BODY-25 / COCO-17 / MP-33 (V2)](#wanposeformatconvertv2)
- **WanAnimatePreprocess_V2** (5)
  - [Depth + Pose + Canny Combined (V2)](#depthposecannycombinedv2)
  - [Draw ViT Pose (V2)](#drawvitposev2)
  - [ONNX Detection Model Loader (V2)](#onnxdetectionmodelloaderv2)
  - [Pose and Face Detection (V2)](#poseandfacedetectionv2)
  - [Wan-Animate Face Quality Check (V2)](#wananimatefacequalitycheckv2)
- **WanAnimatePreprocess_V2/Gaze** (1)
  - [Wan Iris ControlNet Conditioning (V2)](#waniriscontrolnetv2)
- **WanAnimatePreprocess_V2/KANIBUS** (3)
  - [EAR Blink Detector](#earblinkdetectorc2c)
  - [Pupil Dilation Tracker](#pupildilationtrackerc2c)
  - [Saccade Classifier (300°/s)](#saccadeclassifierc2c)
- **WanAnimatePreprocess_V2/Lighting** (1)
  - [Wan SH Lighting Transfer (V2)](#wanshlightingtransferv2)
- **WanAnimatePreprocess_V2/Quality** (1)
  - [Wan Quality Scorer — Temporal Jitter (V2)](#wanqualityscorerjitterv2)


---

## WanAnimatePreprocess/Pose


### WanPose3DRefineNLFV2

**Shown in the menu as:** Wan Pose 3D Refine — NLF (V2)

Refine ViTPose POSEDATA with NLF 3D pose (jitter/occlusion/temporal) — Wan-Animate ready.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | — |
| `images` | `IMAGE` |  | — |
| `nlf_model` | choice: `nlf_l_multi.torchscript` |  | — |
| `device` | choice: `auto`, `cuda`, `cpu` | default `"auto"` | — |
| `blend_strength` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | How strongly NLF's 3D body overrides ViTPose (1=full NLF). |
| `temporal_smoothing` | `BOOLEAN` | default `True` | — |
| `smoothing_min_cutoff` | `FLOAT` | default `1.0`, range 0.05…10.0, step 0.05 | — |
| `smoothing_beta` | `FLOAT` | default `0.1`, range 0.0…5.0, step 0.01 | — |
| `occlusion_fill` | `BOOLEAN` | default `True` | Where ViTPose confidence is low, trust NLF's 3D projection. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` | — |
| 1 | `info` | `STRING` | — |


---

## WanAnimatePreprocessV2/extras


### WanFaceController3DV2

**Shown in the menu as:** Wan Face Controller 3D

All-in-one face controller for the V2 pose pipeline.

Combines four independent stages, applied in this order:
  (1) Optional reference-shape blend (mouth/brows/eyes/jaw).
  (2) 12 FACS-inspired expression dials via expression_coeffs_json.
  (3) 3-DoF head rotation (yaw/pitch/roll) using a canonical
      iBUG-68 depth map.
  (4) Gaze offset applied to iris_data pupil/iris keypoints.

Leaving a stage's controls at their defaults makes that stage a no-op, so the same node covers any subset.

Schemas: see source file docstring.


**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | Pre-detected pose bundle (iBUG-68 face landmarks). OPTIONAL if you wire `images` + `model` to detect inside this node. |
| `images` | `IMAGE` |  | Video frames (B,H,W,3). Wire a video loader here to run ViTPose pose estimation INSIDE this node — no separate detector node needed. |
| `model` | `POSEMODEL` |  | ViTPose+YOLO bundle from 'ONNX Detection Model Loader (V2)'. Used for internal detection (when `images` is wired and `pose_data` is empty). |
| `detection_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Internal detection: YOLO person-detection confidence threshold. |
| `pose_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Internal detection: per-keypoint confidence threshold. |
| `use_clahe` | `BOOLEAN` | default `True` | Internal detection: CLAHE contrast enhancement on the pose crop. |
| `detect_rescale` | `FLOAT` | default `1.25`, range 1.0…2.0, step 0.05 | Internal detection: bbox padding factor before the ViTPose crop. |
| `fallback_to_full_frame` | `BOOLEAN` | default `True` | Internal detection: run ViTPose on the full frame when YOLO finds no person. |
| `reference_pose_data` | `POSEDATA` |  | Optional single-frame POSEDATA used as a region-wise shape target. |
| `fc3d_config_json` | `STRING` | default `"{"expression_coeffs_json":"","expression_strength":1.0,"expression_clamp":1.5,"expression_clamp_per_axis_json":"","propagate_expression":"off","head_pose_json":"","head_yaw_deg":0.0,"head_pitch_deg":0.0,"head_roll_deg":0.0,"head_tx":0.0,"head_ty":0.0,"head_tz":0.0,"head_scale":1.0,"jaw_rot_deg":0.0,"neck_yaw_deg":0.0,"neck_pitch_deg":0.0,"propagate_head":"off","propagate_gaze":"off","gaze_json":"","gaze_yaw_deg":0.0,"gaze_pitch_deg":0.0,"blend_strength":0.0,"blend_mouth":true,"blend_brows":false,"blend_eyes":false,"blend_jaw":false,"use_metas":"edited","frame_start":-1,"frame_end":-1,"preview_frame_idx":0,"preview_size":512,"preview_max_video_frames":120}"` | Editor-owned JSON for expression/head/gaze/blend/preview params. Synced automatically by face_controller_3d.js. |
| `landmark_overrides_json` | `STRING` | default `""` | JSON {"frames":{"<idx>":{"<lm>":[x_img_norm,y_img_norm]}}} from the in-canvas face viewer (image-normalised 0..1). Empty = no override. |
| `pose_overrides_json` | `STRING` | default `""` | JSON {"frames":{"<idx>":{"<joint>":[x_img_norm,y_img_norm]}}} from the in-canvas pose viewer (OpenPose-18). |
| `gaze_overrides_json` | `STRING` | default `""` | JSON {"frames":{"<idx>":{"l":[yaw_rad,pitch_rad],"r":[yaw_rad,pitch_rad]}}} from the in-canvas gaze handles. |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `unique_id` | `UNIQUE_ID` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` |  |
| 1 | `info` | `STRING` |  |
| 2 | `coeff_time_series_json` | `STRING` |  |
| 3 | `preview_image` | `IMAGE` |  |
| 4 | `overlay_video` | `IMAGE` |  |
| 5 | `keyframes_csv` | `STRING` |  |
| 6 | `pose_diff_json` | `STRING` |  |
| 7 | `lp_rotate_pitch` | `FLOAT` | Head pitch for LivePortrait ExpressionEditor (clamped ±20°). Wire to rotate_pitch. |
| 8 | `lp_rotate_yaw` | `FLOAT` | Head yaw for ExpressionEditor (clamped ±20°). Wire to rotate_yaw. |
| 9 | `lp_rotate_roll` | `FLOAT` | Head roll for ExpressionEditor (clamped ±20°). Wire to rotate_roll. |
| 10 | `lp_pupil_x` | `FLOAT` | Gaze→pupil_x for ExpressionEditor (clamped ±15). Wire to pupil_x. |
| 11 | `lp_pupil_y` | `FLOAT` | Gaze→pupil_y for ExpressionEditor (clamped ±15). Wire to pupil_y. |


### WanGazeETHXGazeV2

**Shown in the menu as:** Wan Gaze — ETH-XGaze Post-Processor (V2)

Replace pose_data['iris_data'] gaze vectors with predictions from the ETH-XGaze ResNet-50 model (ECCV 2020). Requires the pretrained checkpoint epoch_24_ckpt.pth.tar in ComfyUI/models/ethxgaze/.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | POSEDATA bundle from the V2 preprocessor (with iBUG-68 keypoints_face). |
| `images` | `IMAGE` |  | Same RGB image stack the POSEDATA was computed from. Used for the 224x224 face normalisation. |
| `checkpoint` | choice: `<none — drop epoch_24_ckpt.pth.tar in models/ethxgaze/>` |  | ETH-XGaze pretrained weights. Auto-discovered from models/ethxgaze/ and third_party/ETH-XGaze/ckpt/. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `checkpoint_path_override` | `STRING` | default `""` | Absolute path to override the dropdown selection. Empty = use the dropdown. |
| `device` | choice: `auto`, `cuda`, `cpu` | default `"auto"` | — |
| `blend` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.05 | 0 = keep original iris_data gaze, 1 = full ETH-XGaze. Useful to smooth-blend in the new model. |
| `batch_size` | `INT` | default `8`, range 1…64, step 1 | Number of normalised face crops fed through gaze_network per forward pass. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` | — |
| 1 | `info` | `STRING` | — |


### WanPoseDetectViTPoseV2

**Shown in the menu as:** Wan Pose Detect — YOLO + ViTPose (V2)

Standalone YOLO + ViTPose detection. Takes an IMAGE batch and a POSEMODEL bundle (from OnnxDetectionModelLoaderV2) and emits a POSEDATA bundle compatible with the V2 editor and downstream conditioning nodes. No face-cropping / gaze pipeline — keypoints only.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | RGB image stack to detect poses on. Float [0,1], shape (B,H,W,3). |
| `model` | `POSEMODEL` |  | ViTPose+YOLO bundle from OnnxDetectionModelLoaderV2. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `detection_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | YOLO person-detection confidence threshold. |
| `pose_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Per-keypoint confidence threshold; below this the keypoint's confidence is forced to 0. |
| `use_clahe` | `BOOLEAN` | default `True` | CLAHE contrast enhancement on the 256x192 pose crop. Matches the main preprocessor default. |
| `use_blur_for_pose` | `BOOLEAN` | default `False` | Apply a Gaussian blur to the images before YOLO+ViTPose (anti-aliases noisy frames). |
| `blur_radius` | `INT` | default `2`, range 0…32, step 1 | — |
| `blur_sigma` | `FLOAT` | default `1.5`, range 0.0…8.0, step 0.1 | — |
| `rescale` | `FLOAT` | default `1.25`, range 1.0…2.0, step 0.05 | Bbox padding factor before crop for ViTPose. 1.25 = match the main preprocessor. |
| `fallback_to_full_frame` | `BOOLEAN` | default `True` | If YOLO finds no person in a frame, run ViTPose on the entire frame instead of skipping it. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` | — |
| 1 | `info` | `STRING` | — |


### WanPoseFormatConvertV2

**Shown in the menu as:** Wan Pose Format Convert — OP18 → BODY-25 / COCO-17 / MP-33 (V2)

Convert POSEDATA's OpenPose-18 body keypoints to another skeleton format (BODY-25, COCO-17, or MediaPipe-33). Also emits a standard ComfyUI POSE_KEYPOINT JSON so the result plugs directly into OpenPose ControlNet preprocessors.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | POSEDATA bundle from the V2 preprocessor (OpenPose-18 body). |
| `target_format` | choice: `body_25`, `coco_17`, `mediapipe_33`, `openpose_18` | default `"body_25"` | Destination skeleton topology. openpose_18 = passthrough. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `use_metas` | choice: `original`, `edited` | default `"edited"` | Whether to read from pose_metas (edited) or pose_metas_original. |
| `emit_face` | `BOOLEAN` | default `True` | — |
| `emit_hands` | `BOOLEAN` | default `True` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` | — |
| 1 | `pose_keypoint` | `POSE_KEYPOINT` | — |
| 2 | `info` | `STRING` | — |


---

## WanAnimatePreprocess_V2


### DepthPoseCannyCombinedV2

**Shown in the menu as:** Depth + Pose + Canny Combined (V2)

Self-contained ControlNet preprocessor producing depth, pose, canny, normal, layout-combined preview, AND a weighted blended map.

DEPTH backends (set via `depth_backend`):
  - auto       : prefer external_depth_map -> any wired loader -> built_in_midas
  - external   : require external_depth_map IMAGE input
  - built_in_midas : MiDaS small via torch.hub (downloads ~80MB to torch hub cache, no extra node pack needed)
  - damodel_v2     : kijai/ComfyUI-DepthAnythingV2 (models/depthanything/)
  - da3            : PozzettiAndrea/ComfyUI-DepthAnythingV3 (models/depthanything3/) - delegates to V3 pack
  - depthcrafter   : akatz-ai/ComfyUI-DepthCrafter-Nodes (models/depthcrafter/)
  - depth_pro      : spacepxl/ComfyUI-Depth-Pro (models/depth/ml-depth-pro/)

POSE source priority: external_pose_map > posemodel.

NORMAL map: Sobel-from-depth (Lambertian-style RGB). No extra model.

BLEND modes (research-backed, Wikipedia/W3C Compositing 1.0):
  - none           : returns the depth_map
  - weighted_avg   : per-channel sum normalised by total weight (perceptually balanced)
  - screen         : 1 - prod(1 - layer_i*w_i)  (avoids highlight clipping, good for stacking depth+canny gradients)
  - linear_dodge   : min(1, sum(layer_i*w_i))  (additive; sharpens edges; preferred for pose+canny per Fooocus/SDXL controlnet community)
  - max            : per-pixel maximum across weighted layers (preserves strongest cue per pixel)
  - multiply       : prod(layer_i^w_i)  (darkening; emphasises overlap)
  - overlay        : combined multiply/screen S-curve on weighted_avg base
  - channel_split  : R=depth, G=canny, B=pose (Fun-Control / IP-Adapter style multi-condition packing)

OUTPUTS: depth_map, pose_map, canny_map, normal_map, combined_map (layout), blended_map (per blend_mode).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | Input video frames (B,H,W,3) float32 [0,1]. |
| `width` | `INT` | default `832`, range 64…4096 | Output canvas width. |
| `height` | `INT` | default `480`, range 64…4096 | Output canvas height. |
| `enable_depth` | `BOOLEAN` | default `True` | Run the depth pass. Requires at least ONE depth source wired. |
| `enable_pose` | `BOOLEAN` | default `True` | Run the pose pass. |
| `enable_canny` | `BOOLEAN` | default `True` | Run the canny pass. |
| `canny_threshold1` | `INT` | default `100`, range 0…500 | Canny lower hysteresis threshold. |
| `canny_threshold2` | `INT` | default `200`, range 0…500 | Canny upper hysteresis threshold. |
| `canny_aperture` | choice: `3`, `5`, `7` | default `3` | Sobel aperture for Canny (odd: 3/5/7). |
| `depth_colorize` | `BOOLEAN` | default `False` | If true, colorize grayscale depth with INFERNO colormap. Skipped when external_depth_map is already RGB. |
| `depth_invert` | `BOOLEAN` | default `False` | Invert depth (1 - depth). Use when source produces 'far = bright' but you want 'near = bright' (typical ControlNet expectation). |
| `pose_detection_threshold` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.01 | YOLO confidence threshold (only used when posemodel is wired). |
| `pose_draw_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Per-keypoint score threshold for drawing the skeleton. |
| `combined_layout` | choice: `horizontal_3`, `vertical_3`, `grid_2x2`, `depth_only`, `pose_only`, `canny_only` | default `"horizontal_3"` | Layout for the combined output. grid_2x2 = depth \| pose // canny \| original. |
| `depth_backend` | choice: `auto`, `external`, `built_in_midas`, `damodel_v2`, `da3`, `depthcrafter`, `depth_pro` | default `"auto"` | Which depth backend to use. 'auto' tries: external_depth_map -> any wired loader -> built_in_midas. 'built_in_midas' makes the node fully self-contained (downloads MiDaS small via torch.hub on first use, ~80MB). |
| `enable_normal` | `BOOLEAN` | default `True` | Compute Sobel-from-depth NORMAL map. No model required (uses depth pass output). |
| `normal_strength` | `FLOAT` | default `1.0`, range 0.1…10.0, step 0.1 | Scales the Sobel gradients before normalisation. Higher = stronger normal contrast. |
| `blend_mode` | choice: `none`, `weighted_avg`, `screen`, `linear_dodge`, `max`, `multiply`, `overlay`, `channel_split` | default `"weighted_avg"` | How to combine depth+pose+canny+normal into blended_map. linear_dodge=additive (sharp), screen=highlight-safe, channel_split=Fun-Control (R=depth/G=canny/B=pose). |
| `depth_weight` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Weight of depth in blended_map. |
| `pose_weight` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Weight of pose in blended_map. |
| `canny_weight` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Weight of canny in blended_map. |
| `normal_weight` | `FLOAT` | default `0.5`, range 0.0…4.0, step 0.05 | Weight of normal map in blended_map. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `external_depth_map` | `IMAGE` |  | Pre-computed depth IMAGE batch from ANY upstream node. Highest priority. |
| `damodel_v2` | `DAMODEL` |  | DepthAnything V2 model bundle from kijai/ComfyUI-DepthAnythingV2 (DownloadAndLoadDepthAnythingV2Model). Models: ComfyUI/models/depthanything/. |
| `da3_model` | `DA3MODEL` |  | DepthAnything V3 config bundle from PozzettiAndrea/ComfyUI-DepthAnythingV3. Use the V3 pack's Inference node and feed its IMAGE output into external_depth_map. Models: ComfyUI/models/depthanything3/. |
| `depthcrafter_model` | `DEPTHCRAFTER_MODEL` |  | DepthCrafter bundle from akatz-ai/ComfyUI-DepthCrafter-Nodes. Temporally consistent video depth. Models: ComfyUI/models/depthcrafter/. |
| `depth_pro_model` | `DEPTH_PRO_MODEL` |  | Depth-Pro bundle from spacepxl/ComfyUI-Depth-Pro. Metric depth. Models: ComfyUI/models/depth/ml-depth-pro/. |
| `posemodel` | `POSEMODEL` |  | From ONNX Detection Model Loader (V2) or animal-pose loader. Used if enable_pose=True AND no external_pose_map wired. |
| `external_pose_map` | `IMAGE` |  | Pre-rendered pose map from any upstream node (e.g. Fannovel16/comfyui_controlnet_aux DWPose / OpenPose / AnimalPose). Highest priority for pose. |
| `depthcrafter_steps` | `INT` | default `5`, range 1…100 | DepthCrafter only: diffusion inference steps. |
| `depthcrafter_guidance` | `FLOAT` | default `1.0`, range 0.1…10.0, step 0.1 | DepthCrafter only: classifier-free guidance. |
| `depthcrafter_window` | `INT` | default `110`, range 1…200 | DepthCrafter only: temporal window size. |
| `depthcrafter_overlap` | `INT` | default `25`, range 0…100 | DepthCrafter only: window overlap. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `depth_map` | `IMAGE` | Per-frame depth IMAGE batch (3-channel, height x width). |
| 1 | `pose_map` | `IMAGE` | Per-frame pose IMAGE batch (3-channel, on black canvas). |
| 2 | `canny_map` | `IMAGE` | Per-frame canny edge IMAGE batch (3-channel grayscale). |
| 3 | `normal_map` | `IMAGE` | Per-frame normal map (RGB-encoded surface normals from Sobel-of-depth). |
| 4 | `combined_map` | `IMAGE` | Side-by-side combined preview per `combined_layout`. |
| 5 | `blended_map` | `IMAGE` | Weighted blend of {depth, pose, canny, normal} per `blend_mode` and per-channel weights. |


### DrawViTPoseV2

**Shown in the menu as:** Draw ViT Pose (V2)

Render the detected skeleton, face landmarks, iris pupils and gaze arrows onto a clean canvas at the target Wan 2.2 latent resolution. Outputs an IMAGE batch ready to drop into a Wan-Animate sampler.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | From Pose and Face Detection (V2). |
| `width` | `INT` | default `832`, range 64…2048 | Render canvas width (px). Match the sampler latent size. |
| `height` | `INT` | default `480`, range 64…2048 | Render canvas height (px). Match the sampler latent size. |
| `retarget_padding` | `INT` | default `16`, range 0…512 | Padding (px) added around the body bbox when retargeting. Larger = more headroom for big motions. |
| `body_stick_width` | `INT` | default `-1`, range -1…20 | Body skeleton stick width in px. -1 = auto from canvas size. |
| `hand_stick_width` | `INT` | default `-1`, range -1…20 | Hand skeleton stick width in px. -1 = auto. |
| `draw_head` | `BOOLEAN` | default `True` | Draw the head/face skeleton (eyes, nose, ears). |
| `pose_draw_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Per-keypoint score threshold for drawing. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `draw_iris` | `BOOLEAN` | default `True` | Draw iris/pupil markers from MediaPipe iris_data. |
| `draw_gaze` | `BOOLEAN` | default `True` | Draw gaze direction arrows from iris_data. |
| `iris_radius` | `INT` | default `4`, range 1…20 | Pupil circle radius in pixels. |
| `gaze_arrow_len` | `INT` | default `30`, range 4…200 | Length of gaze direction arrow in pixels. |
| `iris_min_confidence` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.01 | Skip iris frames whose detection confidence is below this. |
| `iris_color` | choice: `white`, `magenta`, `yellow`, `green` | default `"white"` | Color of the drawn pupil; magenta gives strongest sampler signal. |
| `face_images` | `IMAGE` |  | OPTIONAL face crop IMAGE batch (typically the face_images_512 output of PoseAndFaceDetectionV2). When wired, the node validates frame-count parity with the pose batch, optionally force-resizes to 512x512, and forwards it on the 'face_video' output so a single DrawViTPoseV2 can feed the Wan-Animate sampler's pose+face inputs in one place. |
| `face_cfg_scale` | `FLOAT` | default `1.0`, range 1.0…10.0, step 0.1, **connection-only** | Passthrough face CFG scale, CONNECTION-ONLY (forceInput) so there is exactly one source of truth: PoseAndFaceDetectionV2.face_cfg_scale. It used to be a second independently-editable widget with the same default, so you could set 2.0 upstream, leave 1.0 here, and get no warning that they had diverged. Unconnected = 1.0 (no-op), which matches the old default. NOTE: Kijai's ComfyUI-WanVideoWrapper has no face-CFG input to wire this into today — for real control over expression adherence use WanVideoAnimateEmbeds.face_strength (spec 2.2's stronger, more direct block-scale lever) instead. |
| `enforce_512_face` | `BOOLEAN` | default `True` | If True and 'face_images' is provided at a non-512 size, force-resize each frame to 512x512 (bilinear) before forwarding. Default True so the encoder always sees the trained input shape. |
| `reference_expression_coeffs_json` | `STRING` | default `""`, multiline | Wan-Animate spec 3.1 (closed-loop critic): wire in the 'expression_coeffs_json' output of a PoseAndFaceDetectionV2 run (export_expression_coeffs=True) on the SOURCE driving video. When non-empty, this node measures ARKit-52 blendshapes from ITS OWN pose_data.iris_data (i.e. the GENERATED Wan-Animate output side, since this node is downstream of the generation pass) and reports per-AU + per-segment error against the reference — a numeric fidelity signal instead of eyeballing frames. Leave empty to skip entirely (zero extra cost). |
| `segment_length` | `INT` | default `77`, range 1…100000 | Frames per segment for the critic's worst-segment breakdown — match WanVideoAnimateEmbeds.frame_window_size (default 77) so segments line up with Wan-Animate's own splice boundaries (spec 2.5/3.5). Only used when reference_expression_coeffs_json is wired. |
| `top_k_aus` | `INT` | default `10`, range 1…52 | How many worst-tracked AUs the critic reports, worst-first. Only used when reference_expression_coeffs_json is wired. |
| `apply_pose_edits_to_face` | choice: `warp`, `off` | default `"warp"` | Expression-edit DELIVERY (2026-07-24). When pose_data carries edited face landmarks (WanFaceController3DV2 expression dials / dragged landmarks) AND face_images is wired, 'warp' moves the ACTUAL face-crop pixels from the original landmark positions to the edited ones (same Delaunay piecewise-affine engine as FC3D's preview), so the Wan-Animate face encoder sees the edit. Without this, landmark edits only change the drawn skeleton — the photographic face crop stays neutral and the sampler follows the crop, i.e. your expression edits silently do nothing. No-op when landmarks are unedited (zero cost), so the default stays 'warp'. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_images` | `IMAGE` | Rendered skeleton IMAGE batch. Feed into your Wan 2.2 Animate sampler. |
| 1 | `face_video` | `IMAGE` | Passthrough face IMAGE batch (512x512 if enforce_512_face). Empty single-frame zero tensor if 'face_images' was not wired. |
| 2 | `face_cfg_scale` | `FLOAT` | Passthrough face_cfg_scale (Wan-Animate paper Sec. 4.3). 1.0 = CFG off. |
| 3 | `critic_report_json` | `STRING` | Wan-Animate spec 3.1 closed-loop critic report (JSON): per-AU mean-absolute-error, per-frame error curve, per-segment breakdown worst-first. '{}' when reference_expression_coeffs_json was not wired. |
| 4 | `worst_aus_csv` | `STRING` | CSV 'name,mae' for the top_k_aus worst-tracked AUs, worst first. Empty string when the critic did not run. |
| 5 | `overall_mae` | `FLOAT` | Mean of all per-AU MAE values (0.0 = perfect match to the reference). 0.0 when the critic did not run. |


### OnnxDetectionModelLoaderV2

**Shown in the menu as:** ONNX Detection Model Loader (V2)

Load ONNX ViTPose + YOLO detection models for Wan 2.2 Animate preprocessing. Place model files in `ComfyUI/models/detection/`. Outputs a `POSEMODEL` bundle that the detection node consumes.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vitpose_model` | choice: `(place .onnx models in ComfyUI/models/detection)` | default `"(place .onnx models in ComfyUI/models/detection)"` | ViTPose ONNX file (human wholebody, e.g. vitpose_h_wholebody_model.onnx). Place in ComfyUI/models/detection/. .onnx is always listed here even if ComfyUI hides it elsewhere. |
| `yolo_model` | choice: `(place .onnx models in ComfyUI/models/detection)` | default `"(place .onnx models in ComfyUI/models/detection)"` | YOLO person-detector ONNX file (e.g. yolov10m.onnx — NOT a pose model). Place in ComfyUI/models/detection/. .onnx always listed. |
| `onnx_device` | choice: `CUDAExecutionProvider`, `CPUExecutionProvider` | default `"CUDAExecutionProvider"` | Execution provider for ONNX Runtime. CUDA is much faster; CPU is the safe fallback. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `POSEMODEL` | ViTPose+YOLO model bundle. Connect to `model` on Pose and Face Detection (V2). |


### PoseAndFaceDetectionV2

**Shown in the menu as:** Pose and Face Detection (V2)

Run YOLO person detection + ViTPose 2D keypoints + (optional) MediaPipe FaceMesh on a video tensor. Produces the full pose/face/iris bundle required by Wan 2.2 Animate Character Replacement workflows.

Wan-Animate fidelity notes (spec 2.5-2.7, workflow-level — not something this node can enforce for you):
2.5 Wan-Animate splices long generations in ~78-frame segments with 1-5 frame temporal handoffs (WanVideoAnimateEmbeds.frame_window_size, default 77); a brief microexpression (often only 2-4 frames) landing exactly on a segment boundary risks being smoothed by the discard/resume splice. If a specific expression must survive, check where it falls relative to your frame_window_size and shift the cut (or duration) if needed.
2.6 Feed this node NATIVE framerate footage. Don't downsample fps upstream (e.g. a LoadVideo 'force_rate' below the source fps) — the face branch's causal 1D-conv temporal downsampling further compresses an already-fps-reduced brief expression, and a 2-4 frame microexpression can be lost entirely before it ever reaches this node.
2.7 If your workflow globally autocasts to fp8/fp16, the face encoder's small-magnitude motion-basis deltas are more vulnerable to quantization noise than the body/pose branch. As of this writing Kijai's ComfyUI-WanVideoWrapper has no per-module precision override for the Wan-Animate face branch specifically — if you need this, load the full checkpoint in fp16/fp32 rather than an fp8 quant.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `POSEMODEL` |  | From ONNX Detection Model Loader (V2). |
| `images` | `IMAGE` |  | Video frames as an IMAGE batch (B,H,W,C float [0,1]). |
| `width` | `INT` | default `832`, range 64…2048 | Target canvas width (px) used for retarget math. Match your Wan 2.2 latent size. |
| `height` | `INT` | default `480`, range 64…2048 | Target canvas height (px). Match your Wan 2.2 latent size. |
| `detection_threshold` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.01 | YOLO confidence threshold. Lower = more permissive person detection. |
| `pose_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Per-keypoint score threshold. Below this a keypoint is treated as missing. |
| `use_clahe` | `BOOLEAN` | default `True` | Apply CLAHE contrast enhancement for pose detection. |
| `use_blur_for_pose` | `BOOLEAN` | default `False` | Apply Gaussian blur internally for YOLO and ViTPose BEFORE detection. Bug-fix (default was True): this softens the exact edges/fine detail ViTPose needs for keypoint precision, producing a visibly blurrier preview and a less accurate skeleton for every user until they discovered and disabled it. Only enable this for genuinely noisy/grainy source footage. |
| `blur_radius` | `INT` | default `5`, range 1…20, step 1 | Gaussian blur kernel radius applied to the face mask edge to soften the boundary. Higher = wider feather. Kernel size = radius*2+1 px. |
| `blur_sigma` | `FLOAT` | default `2.0`, range 0.1…5.0, step 0.1 | Gaussian blur sigma (standard deviation) for the face mask feather. Higher sigma = softer falloff. Tune together with blur_radius. |
| `use_face_smoothing` | `BOOLEAN` | default `True` | Smooth face bounding box center over time. |
| `face_smoothing_strength` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.05 | Higher = more smoothing |
| `use_constant_face_box` | `BOOLEAN` | default `True` | Keep a constant pixel size face crop; position adapts. |
| `face_box_size_px` | `INT` | default `512`, range 64…1024, step 16 | Pixel size of the square face crop when constant mode is on. Default 512 matches Wan 2.2 Animate's face encoder input size; lower values trigger an extra upscale inside the encoder and waste detail. |
| `use_iris_smoothing` | `BOOLEAN` | default `True` | Temporally smooth iris pixel positions across frames. Reduces per-frame jitter that Wan 2.2 Animate's face encoder picks up and reproduces as wobbly gaze. |
| `iris_smoothing_strength` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.05 | EMA mix weight when iris_smoothing_method='ema'. Higher = more smoothing, more lag. Ignored for one_euro / none. |
| `iris_smoothing_method` | choice: `one_euro`, `ema`, `none` | default `"one_euro"` | Iris pixel-position smoother. one_euro = adaptive low-pass (Casiez 2012, recommended). ema = legacy first-order; tweak via iris_smoothing_strength. none = raw per-frame positions. |
| `iris_one_euro_min_cutoff` | `FLOAT` | default `1.0`, range 0.05…10.0, step 0.05 | One-euro min cutoff (Hz) for iris pixel coords. Lower = stronger jitter rejection on near-static eyes (small saccades preserved). |
| `iris_one_euro_beta` | `FLOAT` | default `0.05`, range 0.0…5.0, step 0.01 | One-euro speed coefficient for iris pixel coords. Higher = filter relaxes faster on quick eye movements; lower = stronger steady-state smoothing. |
| `gaze_lock_eyes` | `BOOLEAN` | default `True` | Couple left & right eye gaze so they always look in the SAME direction. Both eyes' yaw/pitch are blended toward their per-frame average. Single most effective fix for the 'eyes pointing different directions' artefact in Wan 2.2 Animate output. |
| `gaze_lock_strength` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | How strongly to pull each eye toward the shared average. 0 = independent (legacy). 1 = perfectly conjugate (both eyes always parallel). 0.7 keeps a touch of natural convergence/divergence. |
| `use_mediapipe_face` | `BOOLEAN` | default `True` | Use MediaPipe FaceMesh (478 pts incl. iris/lips) to override face landmarks. Falls back to ViTPose pupil voting if MediaPipe is missing or fails on a frame. |
| `use_blendshape_gaze` | `BOOLEAN` | default `True` | Use MediaPipe FaceLandmarker (Tasks API) blend shapes for production-grade per-eye yaw/pitch in radians. Head-pose-corrected by training. Auto-downloads face_landmarker.task (~3MB) on first run. Falls back to legacy 2D iris-offset gaze if disabled or unavailable. |
| `gaze_one_euro_min_cutoff` | `FLOAT` | default `1.7`, range 0.05…10.0, step 0.05 | One-euro filter base cutoff frequency (Hz). Lower = more aggressive jitter rejection at the cost of slight lag. 1.7 is a good default for 24-30 fps gaze. |
| `gaze_one_euro_beta` | `FLOAT` | default `0.3`, range 0.0…5.0, step 0.05 | One-euro filter speed coefficient. Higher = filter relaxes faster on quick saccades, preserving responsiveness; lower = stronger smoothing during fast moves. |
| `gaze_max_yaw_deg` | `FLOAT` | default `30.0`, range 5.0…60.0, step 1.0 | Saturation yaw angle in degrees that corresponds to blend shape value 1.0. 30° covers the comfortable physiological range; raise for more dramatic eye motion. |
| `gaze_max_pitch_deg` | `FLOAT` | default `25.0`, range 5.0…60.0, step 1.0 | Saturation pitch angle in degrees that corresponds to blend shape value 1.0. 25° covers the comfortable physiological range. |
| `crop_mode` | choice: `default`, `auto`, `jitterless` | default `"default"` | default = raw detected bbox per frame (NO smoothing / NO constant size — crop is effectively 'off'). auto = legacy smoothed + optional constant-size box. jitterless = TRUE locked crop: every frame is exactly the same size (frame0_size, else face_box_size_px) and the face is held centred within a bounded tolerance even under fast motion — a Mocha-style planar hold. The ONLY thing that changes the size is explicit per-key-frame sizes in keyframes_json. If you instead want the crop to follow the subject's apparent scale (face fills a constant fraction of the tile as they walk toward camera), use 'auto'. |
| `frame0_cx` | `INT` | default `-1`, range -1…8192 | Frame 0 anchor center X in pixels. -1 = use detected face center on frame 0. Used only when crop_mode=jitterless. |
| `frame0_cy` | `INT` | default `-1`, range -1…8192 | Frame 0 anchor center Y in pixels. -1 = use detected face center on frame 0. |
| `frame0_size` | `INT` | default `0`, range 0…4096, step 16 | Locked square crop size in pixels (used for the entire clip). 0 = fall back to face_box_size_px. |
| `keyframes_json` | `STRING` | default `"[]"`, multiline | JSON list of per-frame overrides: [{"frame":N, "cx":X, "cy":Y, "size":S?}, ...]. Frames between key-frames are linearly interpolated. size is optional; if omitted the locked size is kept. |
| `smoothing_method` | choice: `one_euro`, `ema`, `gaussian`, `none` | default `"one_euro"` | Center-trajectory filter. one_euro = jitterless adaptive low-pass (recommended). ema = legacy motion-adaptive EMA. gaussian = fixed-window 1D blur. none = raw. |
| `crop_one_euro_min_cutoff` | `FLOAT` | default `1.0`, range 0.05…10.0, step 0.05 | One-euro min cutoff (Hz) for crop center. Lower = stronger jitter rejection. |
| `crop_one_euro_beta` | `FLOAT` | default `0.05`, range 0.0…5.0, step 0.01 | One-euro speed coefficient for crop center. Higher = filter relaxes faster on quick motion. |
| `crop_gaussian_window` | `INT` | default `7`, range 3…51, step 2 | Window size (odd) for the Gaussian temporal blur of the crop center. |
| `crop_safety_margin` | `FLOAT` | default `1.12`, range 1.0…2.0, step 0.01 | Inflate the crop by this factor before smoothing so filter lag, yaw-foreshortened detections and expression-driven bbox growth cannot clip the face. 1.0 = no margin (old behaviour). Applies to both 'auto' and 'jitterless'. If crop_containment_check reports corrections on more than a handful of frames, raise this toward 1.15-1.20 rather than fighting it downstream. |
| `crop_size_one_euro_beta` | `FLOAT` | default `0.2`, range 0.0…2.0, step 0.01 | One-euro beta for the crop SIZE trajectory, separate from crop_one_euro_beta (which is the CENTER's). Position wants heavy damping to kill detector jitter; scale wants to follow real zoom/approach or the crop under-sizes mid-move. Only used when the size is allowed to vary (crop_mode='auto', or jitterless with explicit key-frame sizes) — a locked jitterless size ignores it by definition. |
| `crop_containment_check` | `BOOLEAN` | default `True` | HARD per-frame guarantee that the actual detected face bbox ends up inside the final crop. After smoothing, any frame whose face escapes the crop is corrected. In 'jitterless' the correction SHIFTS the crop (the exact-size lock is preserved); growing would silently break the lock, so a face genuinely larger than the locked size is reported in the log instead — that means face_box_size_px / crop_safety_margin is too small for the shot. In 'auto' the crop may grow. Correction counts are logged. |
| `crop_containment_tolerance` | `INT` | default `4`, range 0…128 | Extra pixels of slack required around the detected face bbox when crop_containment_check tests containment. |
| `auto_smoothing_method` | choice: `legacy_ema`, `one_euro`, `ema`, `gaussian`, `none` | default `"legacy_ema"` | Which filter crop_mode='auto' uses. 'legacy_ema' keeps auto's original bespoke EMA byte-for-byte (the default, so existing workflows are untouched); the others route auto through the same shared filters jitterless uses, honouring crop_one_euro_* / crop_gaussian_window. Ignored unless crop_mode='auto'. |
| `force_eyes_open` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | Force closed/blinking eyes open. 0 = off (default). 1 = fully open to a natural EAR of ~0.30; intermediate values blend.  This REACHES THE MODEL because it pairs with DrawViTPoseV2's apply_pose_edits_to_face warp: Wan-Animate's face conditioning is 100%% pixel-driven (landmarks only place the crop, the LIA motion encoder reads raw crop pixels), so this node writes opened-eye LANDMARKS and DrawViTPoseV2 warps the actual crop PIXELS to match — using each frame's own crop as the source, so identity, head pose, mouth and lighting are preserved and only the eye aperture changes. Wire face_images/face_images_512 -> DrawViTPoseV2.face_images and leave apply_pose_edits_to_face='warp' (the default) … |
| `eye_open_mode` | choice: `blinks_only`, `all_frames` | default `"blinks_only"` | Which frames force_eyes_open targets. 'blinks_only' = only frames whose measured Eye-Aspect-Ratio falls below eye_open_blink_ear (keeps natural performance, removes blinks). 'all_frames' = whole-shot override, for when the subject squints throughout. |
| `eye_open_blink_ear` | `FLOAT` | default `0.18`, range 0.01…0.4, step 0.01 | Eye-Aspect-Ratio below which a frame counts as a blink for eye_open_mode='blinks_only'. A natural open eye is ~0.28-0.35, a full blink ~0.05-0.15. Raise toward 0.22 to also catch heavy-lidded frames. |
| `eye_align_mode` | choice: `default`, `eye_upper_third` | default `"default"` | Wan-Animate paper recommendation #1: 'eye_upper_third' vertically shifts the face crop so eyes land at the upper third of the 512x512 face encoder input. The encoder reads holistic face appearance, so consistent eye placement directly improves gaze fidelity. 'default' keeps legacy bbox center. |
| `eye_y_fraction` | `FLOAT` | default `0.3`, range 0.1…0.6, step 0.01 | Target eye row as a fraction of crop height (0.30 = upper third). Only used when eye_align_mode = 'eye_upper_third'. |
| `face_cfg_scale` | `FLOAT` | default `1.0`, range 1.0…10.0, step 0.1 | Wan-Animate paper Sec. 4.3 names CFG on the face-conditioning branch as one lever for finer expression control, BUT Kijai's ComfyUI-WanVideoWrapper has no separate face-CFG input to wire this into — wiring it nowhere is a dead passthrough. The wrapper instead exposes a STRONGER, more direct lever for exactly this purpose (spec 2.2: 'a raw face-adapter block-scale... changes contribution before guidance math rather than after'): WanVideoAnimateEmbeds.face_strength (default 1.0, try 1.5-2.5 for stronger expression adherence). Use that widget on your WanVideoAnimateEmbeds node instead. This FLOAT output is kept for any sampler that DOES expose a genuine face-CFG input and for forward-compat; … |
| `gaze_engine` | choice: `l2cs_gaze360`, `l2cs_mpiigaze`, `ethxgaze`, `pose_normalized_resnet50`, `iris_geometric`, `blendshape_head_corrected`, `blendshape_only` | default `"l2cs_gaze360"` | Per-eye gaze yaw/pitch engine. DEFAULT is now l2cs_gaze360 (GPU/CUDA, auto-downloads ~100MB once) so gaze runs on the GPU; blendshape_* are the CPU-only fallbacks.  * iris_geometric (NEW, deterministic): MEASURES the MediaPipe iris centre inside the eye aperture (corner-to-corner, lid-to-lid) instead of estimating gaze with a NN — no per-person appearance bias, per-eye output, blink-gated, composed with the solvePnP head pose + Kalman like blendshape_head_corrected. Best fidelity for animation retargeting (the character's eyeballs copy the performer's iris positions). Pure CPU math, no downloads. * blendshape_head_corrected (DEFAULT, recommended): MediaPipe ARKit blend shapes + solvePnP … |
| `gaze_kalman_meas_std_deg` | `FLOAT` | default `3.0`, range 0.1…20.0, step 0.1 | Kalman measurement noise (degrees). Higher = trust the model less and lean on the velocity model more — smoother. Used by blendshape_head_corrected and l2cs_* engines. |
| `gaze_kalman_process_std` | `FLOAT` | default `0.8`, range 0.05…5.0, step 0.05 | Kalman process noise (rad/s). Roughly the expected saccade velocity scale. Higher = filter reacts faster to genuine motion but jitters more. |
| `gaze_fps` | `FLOAT` | default `30.0`, range 1.0…240.0, step 1.0 | Video fps used by the Kalman dt. Set to match your source clip; affects velocity coupling, not absolute angles. |
| `gaze_calibration_frame` | `INT` | default `-1`, range -1…999999 | W7-G2 per-shot gaze calibration (iris_geometric engine only). Set this to a frame index where the subject looks STRAIGHT AT THE CAMERA; the measured eye-in-head angles on that frame become the zero reference for the whole shot, removing per-person eye-shape bias (the last few degrees of error no model can fix). -1 = off. |
| `apply_gaze_to_face_image` | choice: `off`, `warp`, `overlay`, `replace` | default `"off"` | C0.1: After gaze is computed, optionally deliver the gaze correction into each 512x512 face crop so the face-encoder input visually matches the gaze the sampler will follow. 'off' = leave face_images untouched (default). 'warp' (Wan-Animate spec 1.5, RECOMMENDED over overlay/replace): moves the REAL iris pixels to the gaze-corrected position via a Delaunay piecewise-affine warp (same engine WanFaceController3DV2 uses) instead of painting a synthetic disk — the face encoder's training augmentations were scale/color-jitter/noise, never a hard-edged synthetic object, so a real-pixel warp stays in-distribution. Displacement is clamped to the eye aperture and gated by the same blur check as the … |
| `au_amplify` | `FLOAT` | default `1.0`, range 1.0…1.5, step 0.01 | Wan-Animate spec 2.3: the face encoder compresses to a small fixed-capacity motion-basis vector, so a genuinely subtle real microexpression can sit near the compression noise floor. This pushes each frame's detected face landmarks a bit FURTHER along the direction they already moved from the neutral reference frame (au_amplify_neutral_frame) — amplifying REAL, DETECTED motion so more of it survives compression; it never synthesizes anything that wasn't already measured. 1.0 = off (default). 1.15-1.3 is the range the paper's own architecture analysis suggests; values are capped at 1.5 since the correction is only a 2D (eye-line roll+scale) head-pose approximation, not a full 3D one — the … |
| `au_amplify_neutral_frame` | `INT` | default `0`, range 0…999999 | Frame index to use as the NEUTRAL reference for au_amplify — pick a frame where the subject's expression is relaxed/neutral (Wan-Animate spec 2.4: an already-tense or asymmetric reference eats into the same motion-basis budget the target microexpression needs). Ignored when au_amplify=1.0. |
| `export_expression_coeffs` | `BOOLEAN` | default `False` | Wan-Animate spec 3.1 (closed-loop critic, foundation): export the 'expression_coeffs_json' output — per-frame ARKit-52 blendshapes measured from this run's iris_data. Off by default (no extra cost when unused). Run this node once on the source driving video and once on the Wan-Animate generated output, then wire the source run's expression_coeffs_json into DrawViTPoseV2.reference_expression_coeffs_json for a per-AU fidelity report. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox_override` | `BBOX` |  | Optional external BBOX for the frame-0 anchor. Highest priority; overrides frame0_cx/cy/size widgets. |
| `landmark_overrides_json` | `STRING` | default `"{}"`, multiline | Manual body-keypoint corrections from the Pose editor. Shape: {"<frame>": {"<jointIdx>": [x_px, y_px], ...}, ...} in SOURCE pixels. Written by the viewer's Edit mode — drag a joint to fix a mis-detection and the correction flows through retargeting into pose_data AND the rendered pose images (not just the preview). Leave as {} for pure detection. |
| `retarget_image` | `IMAGE` |  | Optional reference image of the TARGET character. When connected, the detected driver pose is RETARGETED onto this reference's body proportions and position (the same retarget V1 had): the reference's pose is detected, then get_retarget_pose maps the driver's motion onto it. Leave unconnected for straight detection (no retarget). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `pose_data` | `POSEDATA` | Per-frame pose+face+iris dict bundle. Feed into Draw ViT Pose (V2). |
| 1 | `face_images` | `IMAGE` | Cropped face IMAGE batch suitable for face-id encoders. |
| 2 | `key_frame_body_points` | `STRING` | Key-frame body points as JSON string (debug). |
| 3 | `bboxes` | `BBOX` | Per-frame body BBOX list. |
| 4 | `face_bboxes` | `BBOX` | Per-frame face BBOX list. |
| 5 | `iris_data` | `STRING` | Iris/gaze JSON dump (debug). |
| 6 | `debug_image` | `IMAGE` | Annotated debug IMAGE batch (skeleton overlay). |
| 7 | `right_pupil_xy` | `STRING` | Right pupil pixel xy as JSON (per frame). |
| 8 | `left_pupil_xy` | `STRING` | Left pupil pixel xy as JSON (per frame). |
| 9 | `lip_openness_ratio` | `FLOAT` | Mouth-open scalar list (0=closed, 1=wide). |
| 10 | `restore_info` | `FACE_RESTORE_INFO` | Per-frame {x1,y1,x2,y2,size,frame_shape} dict for paste-back nodes. |
| 11 | `face_cfg_scale` | `FLOAT` | CFG scale for the face conditioning input. Wire into the Wan-Animate sampler's face CFG. 1.0 = CFG off (paper default). |
| 12 | `face_images_512` | `IMAGE` | Cropped face IMAGE batch force-resized to 512x512 (bilinear). Pre-shaped for the Wan 2.2 Animate face encoder; wire directly without an extra Resize node. |
| 13 | `expression_coeffs_json` | `STRING` | Wan-Animate spec 3.1: per-frame ARKit-52 blendshapes measured from this run's iris_data, as {fps,names,frames:[{frame,blendshapes}]}. Run this node on BOTH the source driving video and the Wan-Animate GENERATED output, then wire this output from the SOURCE run into DrawViTPoseV2.reference_expression_coeffs_json (with the GENERATED run's pose_data wired into DrawViTPoseV2.pose_data as usual) to get a per-AU fidelity report. Empty '{}' when export_expression_coeffs=False (default) or no MediaPipe … |


### WanAnimateFaceQualityCheckV2

**Shown in the menu as:** Wan-Animate Face Quality Check (V2)

Score each face crop on (a) Laplacian-variance sharpness and (b) eye-region brightness, then optionally repair bad frames by copying the previous good frame or by simple sharpening. Bad face conditioning frames cause the Wan-Animate face encoder to produce drifting / wrong-direction gaze (paper Sec. 4.3). Connect this BETWEEN Pose-and-Face-Detection (V2)'s `face_images` output and your downstream face-id encoder.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `face_images` | `IMAGE` |  | Per-frame 512x512 face crops (output of Pose and Face Detection V2). |
| `blur_threshold` | `FLOAT` | default `50.0`, range 0.0…5000.0, step 1.0 | Laplacian-variance threshold below which a frame is flagged as blurry. Typical sharp 512x512 frames score 100-1000; <50 indicates motion blur or out-of-focus. |
| `min_eye_brightness` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | Minimum mean luma of the eye-region strip (rows 30%-55%). Below this, eyes are likely closed or the frame is too dark for the encoder to read gaze. |
| `auto_repair_bad_frames` | `BOOLEAN` | default `True` | If true, repair frames flagged as bad. If false, just report stats. |
| `repair_strategy` | choice: `copy_previous_good`, `unsharp_mask`, `skip` | default `"copy_previous_good"` | copy_previous_good: replace with last good frame. unsharp_mask: deconvolve-style sharpening. skip: leave untouched but report. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `face_images_repaired` | `IMAGE` | Repaired face IMAGE batch (same shape as input). |
| 1 | `good_frame_ratio` | `FLOAT` | Fraction of frames that passed BOTH thresholds (0..1). |
| 2 | `report_json` | `STRING` | JSON report: per-frame blur score, eye brightness, verdict, repair action. |


---

## WanAnimatePreprocess_V2/Gaze


### WanIrisControlNetV2

**Shown in the menu as:** Wan Iris ControlNet Conditioning (V2)

Render an iris/gaze ControlNet conditioning image from iris_data JSON. Eye masks, iris discs, gaze arrows, gaze-target heatmap.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `iris_data_json` | `STRING` | default `"[]"`, multiline | — |
| `image_width` | `INT` | default `1024`, range 64…8192 | — |
| `image_height` | `INT` | default `1024`, range 64…8192 | — |
| `render_style` | choice: `full`, `iris_only`, `heatmap_only`, `mask_only` | default `"full"` | — |
| `iris_radius_px` | `INT` | default `6`, range 1…80 | — |
| `arrow_scale_px` | `FLOAT` | default `80.0`, range 0.0…400.0, step 5.0 | Pixels of arrow per radian of gaze. |
| `heatmap_sigma_px` | `FLOAT` | default `35.0`, range 1.0…400.0 | — |
| `background` | choice: `black`, `white`, `neutral_grey` | default `"black"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `face_bboxes` | `BBOX` |  | — |
| `reference_image` | `IMAGE` |  | If given, use its (H,W,B) and overlay onto it at low alpha. |
| `overlay_alpha` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `control_image` | `IMAGE` | — |
| 1 | `iris_mask` | `MASK` | — |
| 2 | `info` | `STRING` | — |


---

## WanAnimatePreprocess_V2/KANIBUS


### EARBlinkDetectorC2C

**Shown in the menu as:** EAR Blink Detector


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | — |
| `threshold` | `FLOAT` | default `0.21`, range 0.05…0.5, step 0.005 | — |
| `min_consecutive_frames` | `INT` | default `2`, range 1…30 | — |
| `fps` | `FLOAT` | default `30.0`, range 1.0…240.0, step 0.5 | — |
| `smooth_window` | `INT` | default `3`, range 1…9 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `blink_report_json` | `STRING` | — |
| 1 | `ear_series_json` | `STRING` | — |
| 2 | `blink_count` | `INT` | — |
| 3 | `blink_rate_hz` | `FLOAT` | — |


### PupilDilationTrackerC2C

**Shown in the menu as:** Pupil Dilation Tracker


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | — |
| `normaliser` | choice: `eye_width`, `face_bbox_diag`, `first_frame_radius` | default `"eye_width"` | — |
| `event_threshold` | `FLOAT` | default `1.25`, range 1.0…3.0, step 0.01 | — |
| `min_consecutive_frames` | `INT` | default `3`, range 1…60 | — |
| `smooth_window` | `INT` | default `5`, range 1…31 | — |
| `fps` | `FLOAT` | default `30.0`, range 1.0…240.0, step 0.5 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `face_bboxes` | `BBOX` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `dilation_report_json` | `STRING` | — |
| 1 | `radius_series_json` | `STRING` | — |
| 2 | `mean_normalized` | `FLOAT` | — |
| 3 | `stddev_normalized` | `FLOAT` | — |


### SaccadeClassifierC2C

**Shown in the menu as:** Saccade Classifier (300°/s)


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data` | `POSEDATA` |  | — |
| `fps` | `FLOAT` | default `30.0`, range 1.0…240.0, step 0.5 | — |
| `velocity_threshold_deg_s` | `FLOAT` | default `300.0`, range 30.0…1000.0, step 5.0 | — |
| `min_consecutive_frames` | `INT` | default `1`, range 1…30 | — |
| `one_euro_min_cutoff` | `FLOAT` | default `1.0`, range 0.1…10.0, step 0.1 | — |
| `one_euro_beta` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `saccade_report_json` | `STRING` | — |
| 1 | `velocity_series_json` | `STRING` | — |
| 2 | `saccade_count` | `INT` | — |
| 3 | `saccade_rate_hz` | `FLOAT` | — |


---

## WanAnimatePreprocess_V2/Lighting


### WanSHLightingTransferV2

**Shown in the menu as:** Wan SH Lighting Transfer (V2)

Spherical-harmonics lighting fit (L=2, 9 basis, per RGB channel) with optional relighting onto a target. Basri-Jacobs/Ramamoorthi formulation.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `source_image` | `IMAGE` |  | — |
| `source_normal` | `IMAGE` |  | — |
| `operation` | choice: `fit_only`, `transfer`, `rotate_lights` | default `"fit_only"` | — |
| `rotate_yaw_deg` | `FLOAT` | default `0.0`, range -180.0…180.0, step 1.0 | — |
| `intensity` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `source_albedo` | `IMAGE` |  | — |
| `target_image` | `IMAGE` |  | — |
| `target_normal` | `IMAGE` |  | — |
| `target_albedo` | `IMAGE` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `relit_image` | `IMAGE` | — |
| 1 | `sh_coeffs_json` | `STRING` | — |
| 2 | `shading_map` | `IMAGE` | — |


---

## WanAnimatePreprocess_V2/Quality


### WanQualityScorerJitterV2

**Shown in the menu as:** Wan Quality Scorer — Temporal Jitter (V2)

Per-frame quality and jitter metrics from pose + (optional) expression JSON. Outputs aggregate quality score in [0,1].


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pose_data_json` | `STRING` | default `"[]"`, multiline | — |
| `image_diagonal_px` | `FLOAT` | default `1500.0`, range 64.0…16384.0, step 1.0 | Used to normalise pixel velocities. |
| `confidence_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | — |
| `max_velocity_px` | `FLOAT` | default `60.0`, range 1.0…1000.0, step 1.0 | Body kp px velocity that would yield score 0. |
| `expression_window` | `INT` | default `8`, range 2…60 | — |
| `bad_velocity_thr_px` | `FLOAT` | default `40.0`, range 1.0…1000.0, step 1.0 | — |
| `bad_visibility_thr` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `expression_coeffs_json` | `STRING` | default `""`, multiline | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `metrics_json` | `STRING` | — |
| 1 | `quality_score` | `FLOAT` | — |
| 2 | `mean_body_velocity` | `FLOAT` | — |
| 3 | `mean_face_velocity` | `FLOAT` | — |
| 4 | `mean_expression_jitter` | `FLOAT` | — |
| 5 | `bad_frame_count` | `INT` | — |
