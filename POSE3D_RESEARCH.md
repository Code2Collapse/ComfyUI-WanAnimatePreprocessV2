# 3D Pose for Wan 2.2 Animate — research & plan validation

Date: 2026-06-21. Research-first (no code yet), per request "read the paper and check if my plan works."

## Your plan
> 3D pose estimation (for ~100% accuracy) → convert to a format Wan Animate (and other Wan models) support.

**Verdict: the plan is sound** — and the research confirms both the need and the target format. One refinement: the strongest path here is **not** SMPLest-X.

## What Wan Animate actually consumes (so we target the right format)
From **Wan-Animate** (arXiv:2509.14055): body control is a **skeleton-based representation, extracted with ViTPose**, rendered to **pose frames**, then compressed by **Wan-VAE** and added to the noise latents via a projection layer.
- ⇒ The driving signal is a **ViTPose-keypoint skeleton image sequence** (NOT OpenPose/DWPose). Your requirement is correct and matches the model.
- The paper itself names the weakness we're fixing: skeleton pose has *"spatial ambiguity"* and is *"susceptible to missing or erroneous keypoints."*

## Why raw ViTPose 2D caps accuracy (the gap to close)
Literature is explicit: single-frame ViTPose *"exhibits jitter, occlusion failure, and temporal inconsistency in video"* and *"cannot encode occlusion."* So per-frame 2D is the accuracy ceiling — exactly what a 3D-consistent stage fixes.

## Recommended method — CONFIRMED via SCAIL: use **NLF (NLFPose)**
SCAIL (CVPR 2026, the SOTA character-animation paper) *"employs NLFPose to estimate 3D body keypoints,"* builds a 3D skeleton, renders bones as 3D cylinders (Taichi), and rasterizes to a 2D guidance signal. It **deliberately avoids SMPL** ("identity leakage due to person-specific shape parameters") — reinforcing that SMPLest-X is the wrong tool here.

**Pipeline: NLF (image → 3D keypoints, COCO/wholebody topology) → temporal smooth + occlusion in-fill → reproject to 2D ViTPose-format skeleton frames.**
- **NLF (Neural Localizer Fields)** outputs 3D directly from the image (no SMPL, weights NOT license-gated) and — crucially — its *localizer field* can emit **any joint definition**, so it can produce **ViTPose-wholebody-topology** 3D keypoints natively, which we reproject to the exact 2D format Wan-Animate consumes.
- (A 2D→3D lifter like PoseMamba is the fallback if we want to lift the EXISTING ViTPose 2D rather than run NLF as a second detector.)
- Why this beats SMPLest-X here:
  - **ViTPose-native** (Wan-Animate's exact input); SMPLest-X outputs an SMPL-X mesh that must be converted.
  - **No license-gated weights.** SMPL-X body models are gated (manual download at smpl-x.is.tue.mpg.de); lifters are not.
  - **8 GB-friendly.** Lifters are tiny vs. a whole-body mesh regressor.
  - Directly attacks the documented failures (jitter/occlusion/temporal) via the 3D + temporal prior, then reprojects to stable 2D.

## Cutting-edge reference worth studying
**SCAIL / SCAIL-2** (arXiv:2512.05905, `zai-org/SCAIL-2`): *"studio-grade character animation via in-context learning of **3D-consistent pose representations**."* This is the closest published work to your exact goal — and **SCAIL is already present in the cloned wrappers** (`third_party/.../SCAIL`, `SCAIL2_LOOP_FAST_PATH_PLAN.md`), so we can study its 3D-consistent pose representation directly.

## Honest note on "100% accuracy"
Not literally attainable from monocular video (depth ambiguity is fundamental). Realistic outcome: a **large, measurable** jump in temporal stability + occlusion robustness over raw ViTPose — which is what actually drives Wan-Animate output quality, since the model is sensitive to keypoint jitter/dropouts.

## Proposed build (when approved)
A new node in this pack, downstream of the existing ViTPose detection:
`ViTPose 2D keypoints → 3D lift (PoseMamba/MotionBERT-class) → temporal smooth + occlusion in-fill → reproject to ViTPose-format skeleton frames` → feeds the existing Wan-Animate pose path. Self-contained weights (auto-download), 8 GB-safe.

## Sources
- Wan-Animate — https://arxiv.org/html/2509.14055v1
- SCAIL-2 (3D-consistent pose) — https://arxiv.org/html/2512.05905v1 · https://github.com/zai-org/SCAIL-2
- PoseMamba — https://arxiv.org/html/2408.03540v2
- Monocular 3D HPE survey (2025) — https://www.mdpi.com/1424-8220/25/8/2409
- SOTA leaderboard — https://github.com/Arthur151/SOTA-on-monocular-3D-pose-and-shape-estimation
