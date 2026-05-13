# ComfyUI-WanAnimatePreprocessV2

## Improved Pose, Face & Iris Detection for Wan Video 2.2 Animate

> Based on GitHub Issue **#1410** (WanVideoPreProcessor) and **#10** (WanAnimatePreProcessor)
>
> **Special thanks:** [@kijai](https://github.com/kijai) and [@steven850](https://github.com/steven850)

---

## Table of Contents

1. [Overview](#-overview)
2. [Installation](#-installation)
3. [Model Downloads](#-model-downloads)
4. [Nodes Reference](#-nodes-reference)
   - [ONNX Detection Model Loader (V2)](#1-onnx-detection-model-loader-v2)
   - [Pose and Face Detection (V2)](#2-pose-and-face-detection-v2)
   - [Draw ViT Pose (V2)](#3-draw-vit-pose-v2)
5. [Parameter Deep-Dive](#-parameter-deep-dive)
   - [Detection Threshold](#detection_threshold)
   - [Pose Threshold](#pose_threshold)
   - [use_clahe](#use_clahe)
   - [use_blur_for_pose](#use_blur_for_pose)
   - [blur_radius](#blur_radius)
   - [blur_sigma](#blur_sigma)
   - [use_face_smoothing](#use_face_smoothing)
   - [face_smoothing_strength](#face_smoothing_strength)
   - [use_constant_face_box](#use_constant_face_box)
   - [face_box_size_px](#face_box_size_px)
   - [use_iris_smoothing](#use_iris_smoothing)
   - [iris_smoothing_strength](#iris_smoothing_strength)
   - [Draw ViTPose Parameters](#draw-vitpose-parameters)
6. [Processing Pipeline](#-processing-pipeline)
7. [Output Reference](#-output-reference)
8. [Iris & Gaze Detection](#-iris--gaze-detection)
9. [Debug Image Guide](#-debug-image-guide)
10. [Recommended Settings by Scenario](#-recommended-settings-by-scenario)
11. [Troubleshooting](#-troubleshooting)
12. [FAQ](#-faq)

---

## 📌 Overview

This ComfyUI custom node package provides **production-quality pose and face preprocessing** for the **Wan Video 2.2 Animate** model. It takes input video frames and extracts:

- **Full-body pose data** (body, hands, face keypoints) in the format expected by Wan Animate
- **Stabilised face crops** (512×512) for expression-driven animation
- **Iris/pupil positions and gaze direction** for accurate eye animation
- **Debug visualisation** images showing every detection overlaid on the original frame

### What makes V2 different from V1?

| Feature | V1 (Original) | V2 (This Version) |
|---------|---------------|-------------------|
| Pose stability | Frequent jitter, vanishing limbs | CLAHE + blur preprocessing eliminates jitter |
| Face crops | Raw bbox from keypoints; clips forehead/chin | Constant-size box with temporal smoothing |
| Iris detection | None | Image-based pupil detection with gradient voting |
| Debug output | None | Full diagnostic overlay image |
| Temporal smoothing | None | Motion-adaptive smoothing for face box and iris |
| Lighting robustness | Poor in low light | CLAHE enhancement adapts to any lighting |

---

## 📥 Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `opencv-python`, `onnxruntime` (or `onnxruntime-gpu`), `torch`, `tqdm`.

### Setup

1. Clone or copy this folder into `ComfyUI/custom_nodes/`:
   ```
   ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocessV2/
   ```
2. Download the required models (see below).
3. Restart ComfyUI.
4. The three nodes will appear under the **WanAnimatePreprocess_V2** category.

---

## 📥 Model Downloads

All models go into:
```
ComfyUI/models/detection/
```

### YOLO (Person Detection)

| Model | Link |
|-------|------|
| YOLOv10m (recommended) | [yolov10m.onnx](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx) |

### ViTPose (Keypoint Estimation)

| Model | Size | Link |
|-------|------|------|
| ViTPose-Large (faster) | ~350 MB | [onnx/wholebody](https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody) |
| ViTPose-Huge (more accurate) | ~1 GB | [onnx](https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx) |

> **Note for ViTPose-Huge:** It is split into two files (`vitpose_h_wholebody_model.onnx` + `vitpose_h_wholebody_data.bin`). Both must be in the same folder. Select the `.onnx` file in the model loader.

---

## 🧩 Nodes Reference

### 1. ONNX Detection Model Loader (V2)

**Purpose:** Loads the YOLO and ViTPose ONNX models into memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vitpose_model` | Dropdown | — | Select your ViTPose `.onnx` file from `ComfyUI/models/detection/` |
| `yolo_model` | Dropdown | — | Select your YOLO `.onnx` file from `ComfyUI/models/detection/` |
| `onnx_device` | Dropdown | `CUDAExecutionProvider` | Which device to run inference on |

**Device selection:**
- `CUDAExecutionProvider` — Use your NVIDIA GPU. **Much faster.** Requires `onnxruntime-gpu`.
- `CPUExecutionProvider` — Use CPU only. Works everywhere but significantly slower.

> **Tip:** If you get CUDA errors, ensure your `onnxruntime-gpu` version matches your CUDA toolkit version. You can check with `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`.

**Output:** A `POSEMODEL` object containing both loaded models, to be connected to the **Pose and Face Detection (V2)** node.

---

### 2. Pose and Face Detection (V2)

**Purpose:** The main processing node. Takes input frames and produces pose data, face crops, iris tracking, and debug images.

This is where all the magic happens. See the [Parameter Deep-Dive](#-parameter-deep-dive) section below for detailed explanation of every input.

**Inputs:**

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `model` | POSEMODEL | — | From Model Loader |
| `images` | IMAGE | — | Video frames (B×H×W×3) |
| `width` | INT | 832 | 64–2048 |
| `height` | INT | 480 | 64–2048 |
| `detection_threshold` | FLOAT | 0.05 | 0.0–1.0 |
| `pose_threshold` | FLOAT | 0.3 | 0.0–1.0 |
| `use_clahe` | BOOLEAN | True | — |
| `use_blur_for_pose` | BOOLEAN | True | — |
| `blur_radius` | INT | 5 | 1–20 |
| `blur_sigma` | FLOAT | 2.0 | 0.1–5.0 |
| `use_face_smoothing` | BOOLEAN | True | — |
| `face_smoothing_strength` | FLOAT | 0.6 | 0.0–1.0 |
| `use_constant_face_box` | BOOLEAN | True | — |
| `face_box_size_px` | INT | 224 | 64–1024 |
| `use_iris_smoothing` | BOOLEAN | True | — |
| `iris_smoothing_strength` | FLOAT | 0.4 | 0.0–1.0 |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `pose_data` | POSEDATA | Full pose metadata (body, hands, face, iris) for Wan Animate |
| `face_images` | IMAGE | Stabilised 512×512 face crops, one per frame |
| `key_frame_body_points` | STRING | JSON array of key body points from the first frame |
| `bboxes` | BBOX | Person bounding box from the first frame |
| `face_bboxes` | BBOX | Face bounding box per frame (x1, x2, y1, y2) |
| `iris_data` | STRING | JSON array of per-frame iris positions and gaze vectors |
| `debug_image` | IMAGE | Diagnostic overlay showing landmarks, iris, bboxes |

---

### 3. Draw ViT Pose (V2)

**Purpose:** Renders the detected pose as a coloured skeleton image, suitable as a conditioning input for Wan Animate.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `pose_data` | POSEDATA | — | — | From Pose and Face Detection output |
| `width` | INT | 832 | 64–2048 | Output image width (should match your generation width) |
| `height` | INT | 480 | 64–2048 | Output image height (should match your generation height) |
| `retarget_padding` | INT | 16 | 0–512 | Padding around the pose figure in pixels |
| `body_stick_width` | INT | -1 | -1–20 | Width of body bone lines. `-1` = auto-scale |
| `hand_stick_width` | INT | -1 | -1–20 | Width of hand bone lines. `-1` = auto-scale. `0` = don't draw hands |
| `draw_head` | BOOLEAN | True | — | Whether to draw the head circle/arc |
| `pose_draw_threshold` | FLOAT | 0.3 | 0.0–1.0 | Confidence threshold for drawing joints |

**Output:** `pose_images` — A batch of skeleton images (B×H×W×3) as float tensors.

---

## 🔧 Parameter Deep-Dive

### `detection_threshold`

**What it does:** Controls the minimum confidence score for YOLO person detections to be accepted.

**How it works:** YOLO outputs bounding boxes with confidence scores. Any detection below this threshold is discarded. This is applied during YOLO post-processing before pose estimation begins.

**How to use:**

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `0.01–0.05` | Accept almost all detections | Partial occlusion, person at edge of frame, small person in wide shot |
| `0.05` (default) | Balanced | Most videos |
| `0.1–0.3` | Strict filtering | Clean, well-lit single-person videos. Reduces false positives |
| `0.5+` | Very strict | Only use if you're getting phantom detections from background objects |

**Best practice:** Start with `0.05`. Only increase if you see YOLO detecting non-human objects (chairs, mannequins, etc.). Decrease if the person occasionally disappears, especially in frames where they're partially off-screen.

> ⚠️ Setting this too high causes the person to "flicker" in and out of detection across frames.

---

### `pose_threshold`

**What it does:** Zeroes out any keypoint with confidence below this value. Affects both the stored pose data and drawn skeletons.

**How it works:** After ViTPose predicts all 133 keypoints (body + face + hands), any keypoint with confidence < `pose_threshold` gets its confidence set to 0. This means:
- The keypoint won't be drawn by Draw ViTPose
- The keypoint won't contribute to face bbox calculations
- The keypoint will still exist in the data but marked as "not visible"

**How to use:**

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `0.0` | Keep all keypoints regardless of confidence | Maximum data retention; useful if downstream processing handles filtering |
| `0.1–0.2` | Very lenient | Difficult poses, people facing away, occluded limbs |
| `0.3` (default) | Balanced | Standard forward-facing or side-view footage |
| `0.5–0.7` | Strict | Only want highly confident keypoints. May cause limbs to disappear |
| `0.8+` | Extreme | Not recommended. Most keypoints will be filtered out |

**Best practice:** `0.3` works for most content. Lower to `0.1` if limbs keep vanishing. The pose draw node has its own separate `pose_draw_threshold` that controls what gets rendered visually.

---

### `use_clahe`

**What it does:** Enables CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the cropped image fed to ViTPose.

**How it works:** CLAHE divides the image into small tiles and equalises the histogram in each tile independently, then blends the boundaries. This dramatically improves local contrast without blowing out already-bright areas. It helps ViTPose "see" keypoints in challenging lighting.

**When to enable (True):**
- Low-light or night-time footage
- Subject wearing dark clothing against a dark background
- Subject wearing white clothing against a bright background
- Indoor scenes with mixed lighting (window + overhead)
- Any video where pose detection is inconsistent

**When to disable (False):**
- Already well-lit, high-contrast studio footage
- If you notice artifacts in the pose output (very rare)
- Marginally faster without it (saving ~1ms per frame)

**Best practice:** Leave it **True**. It helps in nearly all situations and the overhead is negligible. The enhancement is only applied to the detection pipeline — your face crops and output images remain unaffected.

---

### `use_blur_for_pose`

**What it does:** Applies a Gaussian blur to the frames before they're fed to YOLO (person detection) and ViTPose (keypoint estimation).

**How it works:** A Gaussian kernel smooths high-frequency details (fabric texture, hair strands, skin pores, compression artifacts). Pose models are trained to detect joint *positions*, not texture — blurring aligns the input distribution closer to what the model expects.

**Why it helps:**
- Removes compression noise (especially from H.264/H.265 video)
- Eliminates micro-texture that causes multiple weak heatmap peaks
- Produces cleaner, more confident keypoint predictions
- Stabilises YOLO bounding boxes across frames

> **Important:** Blur is only applied to the detection pipeline. Face crops come from the original unblurred frames.

**When to enable (True):**
- Video input (any codec/compression)
- Images with visible noise or grain
- Textured clothing or busy backgrounds

**When to disable (False):**
- High-resolution studio photography with no noise
- Already-downscaled/blurry input
- If you want to experiment and compare results

**Best practice:** Leave it **True**. The improvement is substantial and consistent.

---

### `blur_radius`

**What it does:** Controls the size of the Gaussian blur kernel used for pose detection.

**How it works:** The actual OpenCV kernel size is `blur_radius * 2 + 1`. So `blur_radius=5` produces an 11×11 kernel. Larger kernels produce more smoothing.

| blur_radius | Kernel size | Effect |
|-------------|-------------|--------|
| 1 | 3×3 | Minimal smoothing |
| 3 | 7×7 | Light smoothing |
| **5** (default) | **11×11** | **Good balance** |
| 7 | 15×15 | Strong smoothing |
| 10+ | 21×21+ | Heavy smoothing; may lose small-joint detail |

**Best practice:** `5` is optimal for 720p–1080p video. For 4K input you might increase to `7`. For low-resolution (480p or below), try `3`.

> ⚠️ Extremely high values (>10) can blur away hand and face keypoints.

---

### `blur_sigma`

**What it does:** Controls the "spread" of the Gaussian blur. Higher sigma = softer, wider blur.

**How it works:** Sigma (σ) determines how much each pixel's neighbours influence it. The kernel size sets the maximum range; sigma sets how quickly the influence drops off. A large sigma relative to the kernel size makes the blur uniform across the kernel.

| blur_sigma | Effect |
|------------|--------|
| 0.5–1.0 | Barely visible; only removes very fine noise |
| **2.0** (default) | **Removes texture noise while preserving body shape** |
| 3.0–4.0 | Stronger smoothing; good for very noisy input |
| 5.0 | Maximum; significant detail loss |

**Best practice:** `2.0` with `blur_radius=5` is the sweet spot for standard video. If your input is very grainy (e.g., ISO 3200+ phone footage at night), try `3.0`.

---

### `use_face_smoothing`

**What it does:** Enables temporal smoothing of the face bounding box centre across frames.

**How it works:** Without smoothing, the face bbox jitters frame-to-frame because ViTPose keypoints have small per-frame variations. This smoothing applies a **motion-adaptive exponential filter** to the face box centre:
- When the face is still → heavy smoothing (nearly static)
- When the face moves slightly → moderate smoothing
- When the face moves quickly (turning, nodding) → minimal smoothing (immediate tracking)

This prevents the generated face crops from jittering while still tracking rapid motion.

**When to enable (True):**
- Any video where the subject's face appears across multiple frames
- Talking-head videos
- Dance/movement videos

**When to disable (False):**
- Single-image processing (no temporal context)
- Rapidly switching between different subjects
- If you want raw detection output for further processing

**Best practice:** Leave it **True** for video. The motion-adaptive behaviour ensures it never lags behind real movement.

---

### `face_smoothing_strength`

**What it does:** Controls how much temporal smoothing is applied to the face box centre. Only active when `use_face_smoothing` is True.

**How it works:** This is the "base" smoothing strength. The actual per-frame smoothing is dynamically reduced when motion is detected. `0.0` means no smoothing; `1.0` means maximum smoothing (when the face is stationary).

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `0.0` | No smoothing (same as disabling) | If you want raw bbox per frame |
| `0.2–0.3` | Light smoothing | Fast-moving subjects; sports footage |
| `0.4–0.6` | Moderate smoothing | Standard talking-head content |
| **`0.6`** (default) | **Good balance** | **Most videos** |
| `0.8–1.0` | Heavy smoothing | Very stable face (news anchor, interview) |

**Best practice:** `0.6` works well for most content. Increase to `0.8` for interview-style footage where the face barely moves. Decrease to `0.3` for action sequences.

---

### `use_constant_face_box`

**What it does:** Locks the face crop to a fixed pixel size instead of computing a variable-size bounding box from keypoints each frame.

**How it works:** ViTPose only detects eye, nose, and mouth landmarks — it has **no forehead or chin keypoints**. This means a bbox computed from keypoints alone will:
- Clip the forehead (especially with hair up or hats)
- Miss the chin in downward head tilts
- Change size erratically when keypoints move

With constant face box enabled, the system:
1. Computes the face bbox centre from keypoints
2. Applies temporal smoothing to that centre
3. Creates a square box of exactly `face_box_size_px` centred on the smoothed point

The result is a perfectly stable zoom level where only the centre moves.

**When to enable (True):**
- Standard Wan Animate workflows (recommended)
- Any scenario where you want consistent face framing across frames
- Preventing sudden zoom-in/zoom-out in face crops

**When to disable (False):**
- If the subject distance changes dramatically (walking toward/away from camera)
- Multi-person scenes where you want the bbox to adapt to different face sizes
- Creative workflows where variable framing is desired

**Best practice:** Leave it **True** for Wan Animate. The consistent framing produces much better animation results.

---

### `face_box_size_px`

**What it does:** Sets the pixel size of the square face crop region (before the final 512×512 resize). Only applies when `use_constant_face_box` is True.

**How it works:** This is essentially a **zoom level** control. The face region is always cropped as a square of this size from the original frame, then resized to 512×512 for the Animate model.

- **Smaller value** = more zoomed in (only eyes/nose/mouth visible)
- **Larger value** = more zoomed out (full head + shoulders visible)

**Size guide based on framing:**

| Shot Type | Subject Coverage | Recommended Size | Notes |
|-----------|-----------------|-----------------|-------|
| Extreme close-up | Face fills >70% of frame | `320–448` | Preserves forehead and chin |
| Close-up | Head and shoulders | `256–320` | Good default range |
| **Medium shot** (chest up) | **Most common** | **`224–256`** | **Start here** |
| Medium-wide (waist up) | Half body | `192–256` | Still captures head turns |
| Full body | Entire person | `128–192` | Small face; may lose detail |
| Wide shot | Person is small | `96–128` | May need to increase |

**How to find the right value:**
1. Start with `224` (default)
2. Run the workflow and check the `face_images` output
3. If the forehead or chin is clipped → increase by 32–64
4. If there's too much background in the face crop → decrease by 32–64
5. Check that the ears are visible for profile views

> **Tip:** The output face crop is always 512×512 regardless of this setting. A smaller `face_box_size_px` means the face is upscaled more, which can introduce blur. Keep it large enough to capture the full head at native resolution.

---

### `use_iris_smoothing`

**What it does:** Enables temporal smoothing of detected iris/pupil positions across video frames.

**How it works:** Without smoothing, the detected pupil centre can jump slightly between frames due to image noise, lighting changes, or slight inaccuracies in the detection pipeline. Iris smoothing applies an exponential moving average to the (x, y) iris positions:

```
smoothed_pos = alpha × current_pos + (1 - alpha) × previous_smoothed_pos
```

where `alpha = 1.0 - iris_smoothing_strength`.

**When to enable (True):**
- Video sequences where eye gaze should appear smooth
- Talking-head videos where the subject looks at the camera
- Any footage where you're using iris/gaze data for animation

**When to disable (False):**
- Single-image processing
- Footage with very rapid eye movements you want to preserve (rare)
- If you don't use the iris_data output at all

**Best practice:** Leave it **True** for video. Eye movement is naturally smooth in real life, so temporal smoothing produces more realistic results.

---

### `iris_smoothing_strength`

**What it does:** Controls the amount of temporal smoothing applied to iris positions when `use_iris_smoothing` is True.

**How it works:** Higher values smooth more aggressively. The filter is applied independently to left and right iris positions.

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `0.0` | No smoothing | If you want raw detection per frame |
| `0.1–0.2` | Very light | Fast eye movements that need to be preserved |
| `0.3–0.4` | Moderate | Most standard footage |
| **`0.4`** (default) | **Good balance** | **Standard talking-head videos** |
| `0.6–0.8` | Strong smoothing | Subject looking steadily at camera; minimises jitter |
| `1.0` | Maximum (iris barely moves) | Not recommended for natural results |

**Best practice:** `0.4` for most content. Increase to `0.6` if you notice iris jitter in the debug image. Keep below `0.7` to preserve natural eye movement.

---

### Draw ViTPose Parameters

#### `retarget_padding`

**What it does:** Adds padding (in pixels) around the entire pose figure on the output canvas.

**How it works:** Before drawing the skeleton, the pose is retargeted to fit within the output dimensions minus this padding on each side. More padding = smaller skeleton centred on the canvas.

| Value | Effect |
|-------|--------|
| `0` | Skeleton fills the entire canvas |
| `16` (default) | Small margin; good for most cases |
| `32–64` | Noticeable margin; prevents limbs touching edges |
| `128+` | Large margin; tiny skeleton in the centre |

**Best practice:** `16` is fine for Wan Animate. Increase if limbs get cut off at the frame edges.

#### `body_stick_width`

**What it does:** Width (in pixels) of the lines connecting body joints.

| Value | Effect |
|-------|--------|
| `-1` (default) | Auto-calculated based on output resolution |
| `0` | Don't draw body at all |
| `1–5` | Thin lines |
| `6–10` | Medium (good for 720p–1080p) |
| `11–20` | Thick lines (good for 4K or emphasis) |

#### `hand_stick_width`

**What it does:** Width (in pixels) of lines connecting hand/finger joints.

| Value | Effect |
|-------|--------|
| `-1` (default) | Auto-calculated |
| `0` | **Don't draw hands at all** |
| `1–3` | Thin (hands are small; thin lines work well) |

> **Tip:** Set to `0` if hand tracking is unreliable or not needed.

#### `draw_head`

**What it does:** Whether to draw the head circle/arc on the pose image.

- `True` (default): Draws a head representation based on face keypoints
- `False`: Omits the head; useful if you only need body pose

#### `pose_draw_threshold`

**What it does:** Controls which keypoints/bones are drawn based on their confidence.

This is separate from `pose_threshold` — the detection node stores all keypoints, while this threshold controls which ones are *rendered* as visual output.

| Value | Effect |
|-------|--------|
| `0.0` | Draw everything, even uncertain joints (may look messy) |
| `0.05–0.1` | Draw most joints; prevents fully random points from showing |
| `0.2–0.3` | Draw confident joints; some minor limbs may disappear |
| **`0.3`** (default) | **Balanced; matches detection threshold** |
| `0.5+` | Only very confident joints; may lose fingers, toes, obscured limbs |

**Best practice:** Use `0.3` for clean results. Lower to `0.05` if limbs keep vanishing in the pose output.

---

## 🔄 Processing Pipeline

```
Input Frames (B × H × W × 3)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: PREPROCESSING        │
│  • Optional CLAHE enhancement   │
│  • Optional Gaussian blur       │
│  → Blurred frames (for detect)  │
│  → Original frames (for crops)  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2: YOLO PERSON DETECT   │
│  • Runs on blurred frames       │
│  • Outputs person bounding box  │
│  • Filters by detection_thresh  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 3: ViTPose INFERENCE     │
│  • Crops person from blurred    │
│  • CLAHE on crop (optional)     │
│  • Normalises input             │
│  • Predicts 133 keypoints:      │
│    23 body + 21 LH + 21 RH     │
│    + 68 face landmarks          │
│  • Filters by pose_threshold    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 4: POSE META EXTRACTION  │
│  • Normalise keypoints to [0,1] │
│  • Split into body/face/hands   │
│  • Build pose meta structures   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 5: FACE BBOX BUILD       │
│  • Compute raw bbox from face   │
│    keypoints (scale=1.3)        │
│  • Extract centres              │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 6: FACE TEMPORAL SMOOTH  │
│  • Motion-adaptive exponential  │
│    smoothing of bbox centres    │
│  • Constant-size box from       │
│    smoothed centre              │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 7: FACE CROPPING         │
│  • Crop from ORIGINAL frames    │
│  • Resize to 512×512            │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 8: IRIS / GAZE DETECT    │
│  • Extract eye ROI per frame    │
│  • CLAHE + upper-region mask    │
│  • Gradient voting (primary)    │
│  • Contour analysis (secondary) │
│  • Temporal smoothing            │
│  • Compute gaze vectors         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 9: DEBUG VISUALISATION   │
│  • Draw all 68 face landmarks   │
│  • Eye contour polylines        │
│  • Iris crosshairs + confidence │
│  • Gaze direction arrows        │
│  • Face and body bounding boxes │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 10: PACKAGING            │
│  • PoseData with iris metadata  │
│  • face_images tensor           │
│  • JSON outputs (keypoints,     │
│    iris/gaze data)              │
│  • Debug image tensor           │
└─────────────────────────────────┘
```

---

## 📦 Output Reference

| Output | Type | Format | Usage |
|--------|------|--------|-------|
| `pose_data` | POSEDATA | Dict with `pose_metas`, `pose_metas_original`, `iris_data` | Connect to Draw ViTPose and/or Wan Animate nodes |
| `face_images` | IMAGE | Float32 tensor [B, 512, 512, 3] | Connect to Wan Animate face input |
| `key_frame_body_points` | STRING | JSON: `[{"x": 123, "y": 456}, ...]` | 8 key body points from frame 0 for reference |
| `bboxes` | BBOX | Tuple of 4 ints (x1, y1, x2, y2) | Person bounding box from frame 0 |
| `face_bboxes` | BBOX | List of (x1, x2, y1, y2) per frame | Face crop regions used |
| `iris_data` | STRING | JSON array (see below) | Per-frame iris + gaze |
| `debug_image` | IMAGE | Float32 tensor [B, H, W, 3] | Connect to Preview Image for debugging |

### Iris Data JSON Format

```json
[
  {
    "frame": 0,
    "right_iris": {"x": 234.5, "y": 167.2, "confidence": 0.85},
    "left_iris": {"x": 345.1, "y": 165.8, "confidence": 0.82},
    "right_gaze": {"dx": 0.1234, "dy": -0.0567},
    "left_gaze": {"dx": 0.0987, "dy": -0.0432}
  },
  ...
]
```

- `x`, `y` — Pixel coordinates of the detected pupil centre
- `confidence` — Detection confidence (0.0–1.0). Below 0.1 indicates fallback to geometric centre
- `dx`, `dy` — Normalised gaze direction unit vector (iris offset from geometric eye centre)
  - `dx > 0` → looking right; `dx < 0` → looking left
  - `dy > 0` → looking down; `dy < 0` → looking up
  - Magnitude near 0 → looking straight ahead

---

## 👁️ Iris & Gaze Detection

### How It Works

The iris detection pipeline uses a **multi-strategy approach** to robustly locate the pupil centre within each eye:

#### Strategy 1: Gradient-Based Centre Voting (Primary)

Inspired by the **Timm-Barth (2011)** algorithm. Edge gradients around the circular pupil boundary naturally point *radially outward*. By casting rays in the negative gradient direction from every strong-edge pixel, votes accumulate at the true pupil centre.

**Advantages:**
- Threshold-free — doesn't depend on absolute pixel darkness
- Robust to varying lighting, iris colour, and reflections
- Naturally finds circular features (the pupil)

#### Strategy 2: Multi-Threshold Contour Analysis (Secondary)

Tries multiple intensity percentiles (10th, 20th, 30th, 40th) to binarise the eye region, then evaluates each dark blob by:
- **Circularity** — pupil is round
- **Proximity** — pupil is near the geometric eye centre (with asymmetric vertical penalty to avoid eyelid shadows)
- **Size** — pupil is 8–45% of the visible eye area

#### Strategy 3: Weighted Dark-Pixel Centroid (Tertiary)

Computes the weighted centroid of all dark pixels within the eye mask, biased toward the upper region of the eye opening.

#### Key Improvements Over Naive Approaches

1. **Upper-region restriction:** The search mask is limited to the upper 65% of the eye opening. The pupil is biologically at or above the vertical midpoint — lower regions contain eyelid shadows and eyelash darkness that contaminate detection.

2. **CLAHE preprocessing:** Local histogram equalisation on the eye ROI dramatically improves pupil-iris-sclera contrast, especially under uneven lighting.

3. **Eye Aspect Ratio (EAR) check:** Closed or nearly-closed eyes (EAR < 0.12) are detected early and assigned the geometric centre with low confidence, avoiding false detections on eyelid creases.

4. **Asymmetric vertical scoring:** Candidates below the geometric eye centre are penalised more heavily than those above, reflecting the natural anatomy where the pupil centre is typically at or slightly above the midline of the visible eye opening.

---

## 🔍 Debug Image Guide

The `debug_image` output shows annotations overlaid on the original frame:

| Element | Colour | Description |
|---------|--------|-------------|
| **Jawline dots** | Yellow `(255, 200, 0)` | Landmarks 1–17 |
| **Eyebrow dots** | Yellow-Green `(200, 255, 0)` | Landmarks 18–27 |
| **Nose dots** | Blue `(0, 0, 255)` | Landmarks 28–36 |
| **Eye dots** | Green `(0, 255, 0)` | Landmarks 37–48 |
| **Outer mouth dots** | Cyan `(0, 255, 255)` | Landmarks 49–60 |
| **Inner mouth dots** | Teal `(0, 255, 200)` | Landmarks 61–68 |
| **Eye contour polylines** | Green | Closed polygon connecting 6 eye contour points |
| **Iris crosshair** | Magenta `(255, 0, 255)` | Cross marker at detected pupil centre |
| **Iris circle** | Magenta | Small circle around detected pupil |
| **Confidence number** | Magenta | Iris detection confidence (0.00–1.00) |
| **Gaze arrow** | Orange `(0, 200, 255)` | Arrow from iris centre showing gaze direction |
| **Face bounding box** | Yellow `(255, 255, 0)` | "FACE" labelled rectangle |
| **Body bounding box** | Green `(0, 255, 0)` | "BODY" labelled rectangle |

### How to Read the Debug Image

1. **Landmarks correctly placed?** Each coloured dot should sit on the corresponding facial feature. If dots are wildly off, check your `detection_threshold` and `pose_threshold`.

2. **Eye contour reasonable?** The green polygon should outline each eye. If it's skewed, the ViTPose detection may be struggling — try enabling CLAHE and blur.

3. **Iris crosshair on the pupil?** The magenta cross should be centred on the dark pupil. If it's offset, the confidence number tells you how sure the detector is.

4. **Gaze arrow direction makes sense?** For a person looking at the camera, arrows should be very short (near zero). For a person looking left, arrows point left. If arrows consistently point the wrong way, the iris detection may need different parameters.

5. **Face box captures the full head?** The yellow rectangle should include forehead and chin. Adjust `face_box_size_px` if it's too tight or too loose.

---

## 🎯 Recommended Settings by Scenario

### Standard Talking-Head Video (Most Common)

```
detection_threshold: 0.05
pose_threshold:      0.3
use_clahe:           True
use_blur_for_pose:   True
blur_radius:         5
blur_sigma:          2.0
use_face_smoothing:  True
face_smoothing_strength: 0.6
use_constant_face_box:   True
face_box_size_px:        224
use_iris_smoothing:      True
iris_smoothing_strength: 0.4
```

### Action / Dance Video (Lots of Movement)

```
detection_threshold: 0.03
pose_threshold:      0.2
use_clahe:           True
use_blur_for_pose:   True
blur_radius:         5
blur_sigma:          2.0
use_face_smoothing:  True
face_smoothing_strength: 0.3    ← less smoothing for fast movement
use_constant_face_box:   True
face_box_size_px:        256    ← slightly larger to handle tilts
use_iris_smoothing:      True
iris_smoothing_strength: 0.2    ← preserve quick eye movements
```

### Low-Light / Night Footage

```
detection_threshold: 0.03     ← more lenient for noisy detections
pose_threshold:      0.2
use_clahe:           True     ← critical for dark footage
use_blur_for_pose:   True
blur_radius:         7        ← stronger to counter noise
blur_sigma:          3.0
use_face_smoothing:  True
face_smoothing_strength: 0.7  ← more smoothing (detections noisier)
use_constant_face_box:   True
face_box_size_px:        256
use_iris_smoothing:      True
iris_smoothing_strength: 0.5  ← more smoothing for noisy iris detect
```

### Close-Up Portrait (Face Fills Frame)

```
detection_threshold: 0.05
pose_threshold:      0.3
use_clahe:           True
use_blur_for_pose:   True
blur_radius:         3        ← less blur needed at high resolution
blur_sigma:          1.5
use_face_smoothing:  True
face_smoothing_strength: 0.6
use_constant_face_box:   True
face_box_size_px:        384   ← larger to capture full head
use_iris_smoothing:      True
iris_smoothing_strength: 0.4
```

### Wide Shot (Small Person in Frame)

```
detection_threshold: 0.02     ← very lenient to catch small person
pose_threshold:      0.15     ← keep uncertain keypoints
use_clahe:           True
use_blur_for_pose:   True
blur_radius:         3        ← less blur (person already small)
blur_sigma:          1.5
use_face_smoothing:  True
face_smoothing_strength: 0.5
use_constant_face_box:   True
face_box_size_px:        128   ← small crop since face is small
use_iris_smoothing:      True
iris_smoothing_strength: 0.5
```

### Single Image (No Video Sequence)

```
detection_threshold: 0.05
pose_threshold:      0.3
use_clahe:           True
use_blur_for_pose:   True
blur_radius:         5
blur_sigma:          2.0
use_face_smoothing:  False    ← no temporal context
face_smoothing_strength: 0.0
use_constant_face_box:   True
face_box_size_px:        224
use_iris_smoothing:  False    ← no temporal context
iris_smoothing_strength: 0.0
```

---

## 🛠️ Troubleshooting

### Person Not Detected / Flickering

**Symptom:** YOLO doesn't detect the person in some frames, or detection flickers on/off.

**Fix:**
1. Lower `detection_threshold` to `0.02` or `0.01`
2. Ensure `use_clahe` is True
3. Ensure `use_blur_for_pose` is True with default blur settings
4. Check if the person is very small in frame — YOLO struggles with small subjects

### Limbs/Joints Vanishing

**Symptom:** In the pose output, some limbs disappear intermittently.

**Fix:**
1. Lower `pose_threshold` to `0.15` or `0.1`
2. Lower `pose_draw_threshold` in the Draw ViTPose node to `0.1` or `0.05`
3. Enable `use_clahe` and `use_blur_for_pose`
4. Ensure the person is fully visible and not heavily occluded

### Face Crop Jittering

**Symptom:** Face crops jitter/jump between frames.

**Fix:**
1. Enable `use_face_smoothing` with strength `0.6–0.8`
2. Enable `use_constant_face_box` — this dramatically reduces crop instability
3. If jittering persists, increase `face_smoothing_strength`

### Face Crop Clipping Forehead/Chin

**Symptom:** The 512×512 face crop cuts off the top of the head or the chin.

**Fix:**
1. Increase `face_box_size_px` (try `256`, `320`, or `384`)
2. Enable `use_constant_face_box` to prevent erratic sizing

### Iris Detection Inaccurate

**Symptom:** In the debug image, the magenta iris crosshair is not on the pupil, or the gaze arrow points in a wrong direction.

**Fix:**
1. Ensure `use_clahe` is True (iris detection uses CLAHE internally, but overall image quality affects landmark placement)
2. Check if the eye landmarks (green dots) are correctly placed first — if not, the iris detector has no good reference
3. For low-resolution faces (small or distant subjects), iris detection will be less accurate. Increase `face_box_size_px` or use closer footage
4. Enable `use_iris_smoothing` to smooth out per-frame noise

### ONNX / CUDA Errors

**Symptom:** Errors mentioning `onnxruntime`, CUDA, or DLL issues.

**Fix:**
1. Check `onnx_device` in the Model Loader — switch to `CPUExecutionProvider` if you don't have a compatible GPU
2. Ensure `onnxruntime-gpu` matches your CUDA version: `pip install onnxruntime-gpu==1.17.0` (for CUDA 12.x)
3. If you get "DLL access denied" errors, close all Python processes, then reinstall: `pip install --force-reinstall onnxruntime-gpu`

### Debug Image Shows Wrong Landmarks

**Symptom:** Face landmark dots are placed on wrong locations (e.g., nose landmarks on the ear).

**Fix:**
1. This usually means the ViTPose model is returning poor predictions. Enable both `use_clahe` and `use_blur_for_pose`
2. Ensure you're using a whole-body ViTPose model (not a body-only model)
3. Try using the ViTPose-Huge model for better accuracy

---

## ❓ FAQ

**Q: Do I need all three nodes?**
A: The Model Loader and Pose & Face Detection are required. Draw ViTPose is only needed if you want to generate pose skeleton images for conditioning.

**Q: Can I use this with Wan 2.1 or only 2.2?**
A: This is designed for Wan 2.2 Animate, which expects specific pose meta formats. It may work with other versions but is not guaranteed.

**Q: What resolution should `width` and `height` be?**
A: These should match your target generation resolution. Common values: 832×480, 1280×720, 512×512. The pose data is normalised so it scales to any resolution.

**Q: Why is the blur only applied to detection, not to face crops?**
A: The blur helps pose models detect joints more reliably. But face crops need to be sharp and detailed for the Animate model to read facial expressions. That's why face crops always come from the original unblurred frames.

**Q: How accurate is the iris detection?**
A: It works best when the eye occupies at least 20–30 pixels in width. Close-up to medium shots produce reliable results. For very distant subjects (eye width < 10 pixels), it falls back to the geometric eye centre with low confidence.

**Q: Can I process thousands of frames?**
A: Yes. Memory usage scales linearly with batch size. For very long videos (1000+ frames), process in segments if you run out of VRAM.

**Q: What's the performance like?**
A: On an RTX 3080 with CUDA, expect ~20–30 FPS for detection and pose estimation combined. CPU mode is significantly slower (~2–5 FPS).

---

## 🌗 Results

Tracking quality with V2 shows **significant improvement**:

- ✅ Stable bones — no jitter or vanishing limbs
- ✅ Clean face crops — forehead and chin included
- ✅ Accurate iris/gaze detection — pupils correctly located
- ✅ Robust performance under difficult lighting
- ✅ Smooth temporal transitions across frames

---

## 📄 License

See [LICENSE](LICENSE) file.

> Attached: `nodes.py`, `onnx_models.py` (minor changes to pass thresholds correctly)

---

🙏 **Thanks for all the work on this project.**
This contribution is shared in the hope it can be merged upstream and help improve Animate preprocessing for everyone.

![Example Output](example.png)

## Use in image/video generation pipelines (Flux / Qwen-Image / Wan / Z-Image / ERNIE-VL)

This V2 pack ships three nodes — `OnnxDetectionModelLoaderV2` (loads YOLOv8 + ViTPose ONNX checkpoints into a unified detector bundle), `PoseAndFaceDetectionV2` (per-frame whole-body 133-keypoint detection plus face crop and bbox extraction), and `DrawViTPoseV2` (renders the COCO-WholeBody skeleton with optional face/hand stick widths and head toggles). The `POSEDATA` bundle and rendered `IMAGE` skeleton match the schema the Wan-Animate sampler family expects.

| Model family | Applicability |
|---|---|
| **Wan 2.x Animate (human)** | Native. The skeleton IMAGE from `DrawViTPoseV2` is the human-pose conditioning channel for Wan-Animate; `face_bboxes` / `face_images` from `PoseAndFaceDetectionV2` feed the face conditioning channel. This is the V2 of that pipeline — improved face handling and hand keypoints over V1. |
| **Wan 2.x Animate (animal)** | Not applicable here — use the sister pack `ComfyUI-WanAnimalPreprocess` (AP10k/APT36k animal ViTPose) for quadruped subjects. |
| **Flux** | Indirect. The rendered skeleton is a generic conditioning IMAGE — wire it into a Flux ControlNet pose model when generating human stills with pose control. The face crops are useful as ID/IP-Adapter input. |
| **Qwen-Image** | Indirect. Same pattern as Flux: use the skeleton as ControlNet pose conditioning if a Qwen-compatible pose ControlNet is loaded. |
| **Z-Image** | Indirect. Use the skeleton as a conditioning image where Z-Image supports pose ControlNet. |
| **GLM-Image** | Not currently applicable; GLM-Image does not yet expose a pose ControlNet path. |
| **ERNIE-VL** | Not applicable; ERNIE-VL is a multimodal LLM rather than a diffusion sampler. |

Wiring pattern for the canonical Wan-Animate V2 graph:

```
[Load Video Frames] ──> [Pose and Face Detection V2] ──┬──> POSEDATA ──> [Draw ViT Pose V2] ──> skeleton IMAGE
                                                       ├──> face_bboxes
                                                       └──> face_images
                                                                       │
   [ONNX Detection Model Loader V2] ──> POSEMODEL ──┘                  │
                                                                       ▼
                                                       [Wan-Animate Sampler]
```

Bounding-box outputs are also useful as input to crop / inpaint pipelines (e.g. ComfyUI-CustomNodePacks `Inpaint Crop Pro`) when refining only the body or face region.


---

## Full Node & Parameter Reference

A complete, auto-generated reference covering every node, return type and widget exposed by this pack lives in [NODE_REFERENCE.md](./NODE_REFERENCE.md). It is regenerated whenever `INPUT_TYPES` change.

