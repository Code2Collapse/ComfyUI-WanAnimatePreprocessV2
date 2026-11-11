# Wan-Animate Preprocess V2 — node guide

What each of the 17 nodes is for, whether it earns its place, and how to wire it. Every input and output is listed
in [NODE_REFERENCE.md](NODE_REFERENCE.md), which is generated from the live registration. This guide is the "why and
where".

Review basis (2026-11-12): every node's source was reviewed on 2026-11-03 (CODE2COLLAPSE_MASTER_AUDIT.md). The
defects found there (W1–W14, WP0b) are fixed (ledger L9.120 / L9.121), each with a test that was seen failing first.
Nodes that need model weights were not run with those weights by that review; each section below says so.

## The main chain (Wan 2.2 Animate: character replacement / animation)

```
ONNX Detection Model Loader ─► Pose and Face Detection ─► [edit / refine, optional] ─► Draw ViT Pose ─► Wan-Animate
        (POSEMODEL)               │ pose_data (POSEDATA)                                   pose_images (IMAGE)
                                  └ face_images ─► Face Quality Check ─► face video input of Wan-Animate
```

The optional middle stage takes POSEDATA in and gives POSEDATA out, so these nodes chain in any order: Face Controller
3D, Face/Pose Delta Editor, Gaze ETH-XGaze, Pose 3D Refine NLF.

One limit applies to every face node here. Wan-Animate reads the face through a small face tile, and the pose image
carries no iris. Eye **direction** therefore cannot be steered through these inputs; face crop density and expression
can (measured, see the project notes on the Wan-Animate face ceiling).

## Nodes

### Core: keep, needed by the chain

**ONNX Detection Model Loader (V2)** · `OnnxDetectionModelLoaderV2`
Loads the YOLO person detector and ViTPose (ONNX) from `models/detection/` into one POSEMODEL bundle. The detection
nodes, Face Controller 3D and Depth + Pose + Canny all share it; a session that another node closed is re-opened
(W1 fix). Wire it first, once per workflow.

**Pose and Face Detection (V2)** · `PoseAndFaceDetectionV2`
The main detector: YOLO + ViTPose body keypoints, face landmarks and iris per frame, plus face crops for the face
input. Inputs `width` / `height` = the Wan generation size. Outputs `pose_data` (into the edit stage or Draw ViT Pose)
and `face_images` (into Face Quality Check, then Wan-Animate's face input). Optional FLUX-Kontext retargeting of the
reference only runs with FLUX weights; it now raises an error instead of failing silently (W2–W5).

**Draw ViT Pose (V2)** · `DrawViTPoseV2`
Renders POSEDATA as the pose video Wan-Animate expects, at the generation size: skeleton, face landmarks, and
optionally iris dots and gaze arrows, which only matter to a model that reads them. Skeleton coordinates are mapped
onto the target canvas (W6 fix). Put it last, right before the sampler.

### Editing and refinement (optional, POSEDATA in -> POSEDATA out): keep

**Wan Face Controller 3D** · `WanFaceController3DV2`
The interactive face editor: head pose, expression and the 3D body editor, with keyframes, all written back into
POSEDATA. Use it when the performance needs changes (a different expression, a corrected head turn). Expression edits
reach Wan through Draw ViT Pose's face warp. The heaviest UI in the pack.

**Face/Pose Delta Editor** · `FacePoseDeltaCoreMEC`
Applies landmark edits made in an anchor-relative frame, so an edit follows the head as it moves, with eased
keyframes. Moved here from CustomNodePacks (the old CNP id still loads with `C2C_LEGACY_FACE_POSE_DELTA=1`). Use it for
small, precise landmark fixes; use Face Controller 3D for whole-face changes.

**Wan Gaze — ETH-XGaze Post-Processor (V2)** · `WanGazeETHXGazeV2`
Replaces the gaze vectors in POSEDATA with ETH-XGaze predictions (needs `models/ethxgaze/epoch_24_ckpt.pth.tar`). It
improves the gaze data. Because of the face-tile limit above, it does not steer where Wan-Animate's eyes look. A
residual bias of about 15° on wild footage is a property of the model, not a bug.

**Wan Pose 3D Refine — NLF (V2)** · `WanPose3DRefineNLFV2`
Uses the NLF 3D pose model to fill occluded joints and reduce jitter. Needs NLF weights (`allow_download`, off by
default). Since the W14 fix it refines the poses Draw ViT Pose actually draws. Use it on clips with occlusions or
jittery limbs.

### Quality control: keep (they decide, they do not draw)

**Wan-Animate Face Quality Check (V2)** · `WanAnimateFaceQualityCheckV2`
Scores each face crop for sharpness and eye brightness, and can replace bad frames with the previous good one. Bad
face frames make Wan-Animate's face drift. Wire it between `face_images` and the face input. No defect was found in
review.

**Wan Quality Scorer — Temporal Jitter (V2)** · `WanQualityScorerJitterV2`
Measures per-frame quality and jitter of a pose (and expression) stream, plus an aggregate 0..1 score. Empty input
now scores 0, not 1 (W12 fix). Use it to compare two preprocessing settings on the same clip.

### Eye analysis: keep as analysis tools (reports, not conditioning)

**EAR Blink Detector** · `EARBlinkDetectorC2C`, **Pupil Dilation Tracker** · `PupilDilationTrackerC2C`,
**Saccade Classifier (300°/s)** · `SaccadeClassifierC2C`
Read the eye data that Pose and Face Detection already produced; no extra model. They find blinks (both eyes below
the eye-aspect-ratio threshold for several frames; since L2.34 the more open eye decides, so a wink no longer counts), pupil-size changes normalised by eye width, and saccades. Fixes:
the saccade velocity is measured before smoothing (W8: 450 -> ~80 deg/s on the test clip, i.e. smoothing no longer
hides the jumps); the pupil face box is read the right way round (W9). Use them to choose keyframes, check a
performance, or drive other nodes from the JSON. They do not change the generation by themselves.

### Conditioning for OTHER models: keep (useful outside Wan-Animate)

**Wan Pose Format Convert (V2)** · `WanPoseFormatConvertV2`
Converts the OpenPose-18 body to BODY-25 / COCO-17 / MediaPipe-33 and emits a standard POSE_KEYPOINT, so this pack's
detections feed OpenPose ControlNets and pose editors. The default `use_metas="edited"` path used to empty every
frame; fixed (WP0b).

**Wan Iris ControlNet Conditioning (V2)** · `WanIrisControlNetV2`
Draws eye masks, iris discs, gaze arrows or a gaze-target heatmap from the iris data, as a control image and mask, for
an image/video model with a suitable ControlNet. Since the W10 fix it reads the keys the detector emits. Wan-Animate
does not read it.

**Depth + Pose + Canny Combined (V2)** · `DepthPoseCannyCombinedV2`
A self-contained ControlNet preprocessor: depth, pose, canny, normals, a layout preview and a weighted blend in one
node. Fixes: `pose_draw_threshold` takes effect (W7); the shared detector session is re-opened (W1). It overlaps
general preprocessor packs; it is kept because it reuses this pack's loaded ONNX models and runs without extra
installs.

**Wan SH Lighting Transfer (V2)** · `WanSHLightingTransferV2`
Fits second-order spherical-harmonics lighting from an image + normal map, can rotate it and relight a target. Needs
a normal map. Fixes: yaw rotates about the vertical axis; transfer without a target raises instead of falling through
(W11). Experimental and niche; kept for lighting-matched plates.

## Dense face landmarks from CustomNodePacks

For denser or more precise face landmarks than ViTPose's face points, CustomNodePacks' **Face Landmarks (Facepoints)**
gives 620 points on the GNM head mesh, plus the 68- and 98-point sets and both pupils. Its POSE_KEYPOINT output feeds
OpenPose-style consumers; no node in this pack reads it yet (POSEDATA is this pack's format).
