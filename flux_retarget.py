# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""FLUX.1-Kontext-dev enhanced pose retargeting for PoseAndFaceDetectionV2.

WHY THIS EXISTS — Wan 2.2 Animate's own preprocessing pipeline
(``wan/modules/animate/preprocess/process_pipepline.py``) accepts a
``use_flux`` flag. It is the THIRD retargeting mode the paper supports:

1. No retargeting (``retarget_flag=False``) — driver proportions carry in.
2. Basic retargeting (``retarget_flag=True, use_flux=False``) — reconstruct
   the template pose with the reference character's proportions. This is
   what ``retarget_image`` already wires.
3. Enhanced retargeting (``retarget_flag=True, use_flux=True``) — before
   retargeting, run FLUX.1-Kontext-dev to NORMALIZE the reference image
   and the first template frame to a standard front-facing pose, then
   retarget from that clean neutral pose.

Basic retargeting fails when the reference character is NOT front-facing
(a 3/4 profile, looking down) — the retargeting then carries that head
tilt into the output. FLUX normalizes the pose first, so retargeting starts
from a neutral.

This module ports the two pieces of that flow that are pure logic and have
no other home:

* ``get_editing_prompts`` — choose a FLUX prompt from which limbs are
  visible in the template pose (arm/leg detection from body keypoints).
  Ported verbatim from ``process_pipepline.py`` with the spec intact.
* ``load_flux_kontext`` / ``edit_with_flux`` — lazy-load the
  FLUX.1-Kontext-dev diffusers pipeline and run one image edit.

The re-detection of pose on the EDITED images is done in ``nodes.py`` with
the detector + pose_model that are already loaded there, so this module
stays dependency-light (diffusers + torch only).
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── FLUX pipeline cache ──────────────────────────────────────────────────
# Loading FLUX.1-Kontext-dev is ~30s and several GB of VRAM. Cache one
# pipeline per (path, dtype) so a graph that re-runs doesn't reload it.
_FLUX_CACHE: dict = {}


def load_flux_kontext(path: str, dtype: str = "bfloat16"):
    """Lazy-load the FLUX.1-Kontext-dev pipeline via diffusers.

    Returns a ``FluxKontextPipeline`` (or the Fill pipeline on newer
    diffusers). Cached per (path, dtype). Raises with a clear, actionable
    message if diffusers is missing or the model folder doesn't exist.
    """
    try:
        import torch  # noqa: F401 — needed for the .to(dtype) below
        from diffusers import FluxKontextPipeline
    except ImportError as exc:
        raise ImportError(
            "use_flux needs the diffusers package and the FLUX.1-Kontext-dev "
            "model. Install diffusers into the ComfyUI python with "
            "`python -m pip install diffusers`, then download "
            "FLUX.1-Kontext-dev from "
            "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev "
            "into a folder and point flux_kontext_path at it. "
            f"Original error: {exc}"
        ) from exc

    p = (path or "").strip()
    if not p:
        raise FileNotFoundError(
            "use_flux is on but flux_kontext_path is empty. Download "
            "FLUX.1-Kontext-dev from "
            "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev "
            "and set flux_kontext_path to that folder."
        )
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"flux_kontext_path {p!r} is not a directory. Download "
            "FLUX.1-Kontext-dev from "
            "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev "
            "into a folder and point flux_kontext_path at it."
        )

    import torch
    key = (p, dtype)
    if key in _FLUX_CACHE:
        return _FLUX_CACHE[key]

    dt = getattr(torch, dtype, torch.bfloat16)
    try:
        pipe = FluxKontextPipeline.from_pretrained(p, torch_dtype=dt)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load FLUX.1-Kontext-dev from {p!r}: {exc}. "
            "The folder must be a diffusers checkpoint for "
            "FLUX.1-Kontext-dev (safetensors or bin)."
        ) from exc

    # VRAM STRATEGY (2026-08-13): FLUX.1-Kontext-dev is ~12B params —
    # ~24GB in bf16. Previously this called enable_sequential_offload()
    # AND THEN pipe.to("cuda"), which moved the WHOLE model to VRAM and
    # defeated the offload — that is the "eating a lot of VRAM" the user
    # saw, and it stayed resident because the pipeline is cached for the
    # session. Now pick ONE low-VRAM strategy and do NOT .to("cuda"):
    #   1. enable_model_cpu_offload() — diffusers' recommended low-VRAM
    #      mode (keeps weights on CPU, moves one module to GPU per forward,
    #      ~3-5GB VRAM, reasonable speed). Preferred.
    #   2. enable_sequential_offload() — lowest VRAM (~1-2GB), slowest.
    #   3. .to("cuda") — only if both offloads failed (whole model in VRAM).
    _offloaded = False
    try:
        pipe.enable_model_cpu_offload()
        _offloaded = True
    except Exception as exc:  # noqa: BLE001
        log.warning("PoseAndFaceDetectionV2 [use_flux]: enable_model_cpu_offload "
                     "failed (%s); trying sequential offload.", exc)
        try:
            pipe.enable_sequential_offload()
            _offloaded = True
        except Exception as exc2:  # noqa: BLE001
            log.warning("PoseAndFaceDetectionV2 [use_flux]: sequential offload "
                         "also failed (%s); loading whole model to CUDA.", exc2)
    if not _offloaded and torch.cuda.is_available():
        try:
            pipe = pipe.to("cuda")
        except Exception as exc:  # noqa: BLE001
            log.warning("PoseAndFaceDetectionV2 [use_flux]: could not move "
                         "FLUX pipeline to CUDA (%s); staying on CPU.", exc)

    if len(_FLUX_CACHE) > 2:
        _FLUX_CACHE.clear()
    _FLUX_CACHE[key] = pipe
    return pipe


def free_flux_cache() -> int:
    """Release every cached FLUX pipeline and return VRAM to the session.

    FLUX.1-Kontext-dev is only invoked ONCE per execute (two edits: the
    reference and the first template frame) — it is not per-frame — so
    keeping the 12B model resident for the whole ComfyUI session wastes
    ~24GB for no benefit. Call this after the edits are done. Safe to call
    when nothing is cached. Returns the number of pipelines freed.
    """
    import torch
    n = len(_FLUX_CACHE)
    _FLUX_CACHE.clear()
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return n


def edit_with_flux(pipe, image_np: np.ndarray, prompt: str,
                    guidance_scale: float = 2.5,
                    num_inference_steps: int = 28) -> np.ndarray:
    """Run one FLUX.1-Kontext-dev image edit. Returns an HxWx3 uint8 array.

    The reference pads-resizes the result back to the input H/W
    (``padding_resize``), so the caller gets a same-shape edited frame.

    Wrapped in ``torch.no_grad()`` so the autograd graph for a 12B-model
    inference is not built and held — that graph alone can double the
    VRAM of a forward pass for no benefit (we never train here).
    """
    from PIL import Image
    import torch
    h, w = image_np.shape[:2]
    pil_in = Image.fromarray(image_np.astype(np.uint8))
    with torch.no_grad():
        out = pipe(
            image=pil_in, height=h, width=w, prompt=prompt,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
        ).images[0]
    return np.array(out)


# ── Prompt generation (ported from process_pipepline.get_editing_prompts) ──
def get_editing_prompts(tpl_pose_metas, refer_pose_meta) -> Tuple[str, str]:
    """Choose FLUX prompts from which limbs are visible in the template.

    Ported verbatim from
    ``wan/modules/animate/preprocess/process_pipepline.py::get_editing_prompts``.
    The template is scanned for visible arms/legs (body keypoints 3-7 and
    9-13); the reference and template each get a prompt asking FLUX to
    re-pose the person to a standard front-facing T-pose / A-pose.

    Returns ``(tpl_prompt, refer_prompt)``.
    """
    arm_visible = False
    leg_visible = False
    for tpl_pose_meta in tpl_pose_metas:
        tpl_keypoints = tpl_pose_meta["keypoints_body"]
        if (tpl_keypoints[3].all() != 0 or tpl_keypoints[4].all() != 0
                or tpl_keypoints[6].all() != 0 or tpl_keypoints[7].all() != 0):
            if ((tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75)
                    or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75)
                    or (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75)
                    or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75)):
                arm_visible = True
        if (tpl_keypoints[9].all() != 0 or tpl_keypoints[12].all() != 0
                or tpl_keypoints[10].all() != 0 or tpl_keypoints[13].all() != 0):
            if ((tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75)
                    or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75)
                    or (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75)
                    or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75)):
                leg_visible = True
        if arm_visible and leg_visible:
            break

    def _pose_prompt(meta):
        # Portrait (taller than wide) -> A-pose; landscape -> T-pose.
        if meta["width"] > meta["height"]:
            return ("Change the person to a standard T-pose "
                     "(facing forward with arms extended). "
                     "The person is standing. Feet and Hands are visible in the image.")
        return ("Change the person to a standard pose with the face "
                "oriented forward and arms extending straight down by the "
                "sides. The person is standing. Feet and Hands are visible "
                "in the image.")

    def _arms_prompt(meta):
        if meta["width"] > meta["height"]:
            return ("Change the person to a standard T-pose "
                     "(facing forward with arms extended). "
                     "Hands are visible in the image.")
        return ("Change the person to a standard pose with the face "
                "oriented forward and arms extending straight down by the "
                "sides. Hands are visible in the image.")

    if leg_visible:
        tpl_prompt = _pose_prompt(tpl_pose_meta)
        refer_prompt = _pose_prompt(refer_pose_meta)
    elif arm_visible:
        tpl_prompt = _arms_prompt(tpl_pose_meta)
        refer_prompt = _arms_prompt(refer_pose_meta)
    else:
        tpl_prompt = "Change the person to face forward."
        refer_prompt = "Change the person to face forward."
    return tpl_prompt, refer_prompt
