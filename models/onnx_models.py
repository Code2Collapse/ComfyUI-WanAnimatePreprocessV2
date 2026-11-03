# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Modified 2025 by kijai (Jukka Seppänen) — ComfyUI-WanAnimatePreprocess
#   https://github.com/kijai/ComfyUI-WanAnimatePreprocess (Apache-2.0)
#
# Modified 2025 by steven850 — threshold_conf/reinit/cleanup additions
#   https://github.com/kijai/ComfyUI-WanAnimatePreprocess/issues/10
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np
import torch
import onnxruntime
import logging
import os 
from ..pose_utils.pose2d_utils import box_convert_simple, keypoints_from_heatmaps


def _normalize_onnx_provider(device):
    """MANUAL bug-fix (Apr 2026): normalize common aliases ('cuda', 'cpu',
    'tensorrt') to their official onnxruntime provider names. Previously
    a default of ``device='cuda'`` would propagate through unchanged and
    silently fall back to CPU because 'cuda' is not a valid provider name.
    """
    if not isinstance(device, str):
        return device
    alias = {
        'cuda': 'CUDAExecutionProvider',
        'gpu': 'CUDAExecutionProvider',
        'cpu': 'CPUExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
        'trt': 'TensorrtExecutionProvider',
    }
    return alias.get(device.lower(), device)


class SimpleOnnxInference(object):
    def __init__(self, checkpoint, device='CUDAExecutionProvider', **kwargs):
        # Store initialization parameters for potential reinit
        self.checkpoint = checkpoint
        self.init_kwargs = kwargs

        # MANUAL bug-fix (Apr 2026): normalize device aliases and verify the
        # requested provider is actually available. Previously the session
        # would silently fall back to CPU when 'cuda' was passed (alias bug)
        # or when CUDAExecutionProvider was requested but onnxruntime-cpu
        # was installed instead of onnxruntime-gpu.
        device = _normalize_onnx_provider(device)
        available = onnxruntime.get_available_providers()
        if device not in available:
            logging.getLogger(__name__).warning(
                "Requested ONNX provider %r is not available. Available: %s. "
                "Falling back to CPUExecutionProvider. To enable GPU: "
                "pip install onnxruntime-gpu (and ensure CUDA toolkit matches).",
                device, available,
            )
            device = 'CPUExecutionProvider'

        provider = [device, 'CPUExecutionProvider'] if device == 'CUDAExecutionProvider' else [device]

        self.provider = provider
        self.session = onnxruntime.InferenceSession(checkpoint, providers=provider)

        # Verify the session actually used the requested provider.
        active = self.session.get_providers()
        if device == 'CUDAExecutionProvider' and 'CUDAExecutionProvider' not in active:
            logging.getLogger(__name__).warning(
                "ONNX session for %s requested CUDAExecutionProvider but "
                "onnxruntime selected %s. Inference will run on CPU.",
                os.path.basename(checkpoint), active,
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = self.session.get_inputs()[0].shape[2:]
        self.input_resolution = np.array(self.input_resolution)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_output_names(self):
        output_names = []
        for node in self.session.get_outputs():
            output_names.append(node.name)
        return output_names

    def cleanup(self):
        if hasattr(self, 'session') and self.session is not None:
            # Close the ONNX Runtime session
            del self.session
            self.session = None

    def _live_session(self):
        """The ONNX session, re-opened when a node closed it (A22 L9.120).

        One loader's model bundle is shared by every node wired to it; PoseAndFaceDetectionV2 calls cleanup() after
        its loops, and the next node's forward() then found session=None ('NoneType' object has no attribute 'run')."""
        if getattr(self, "session", None) is None:
            self.reinit()
        return self.session

    def reinit(self, provider=None):
        # Preserve threshold_conf or other settings before reinitialization
        current_threshold_conf = getattr(self, "threshold_conf", 0.05)

        if provider is not None:
            self.provider = provider

        if self.session is None:
            checkpoint = self.checkpoint
            self.session = onnxruntime.InferenceSession(checkpoint, providers=self.provider)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_resolution = self.session.get_inputs()[0].shape[2:]
            self.input_resolution = np.array(self.input_resolution)

        # Restore threshold after reinit
        self.threshold_conf = current_threshold_conf


class Yolo(SimpleOnnxInference):
    def __init__(
        self,
        checkpoint,
        device='cuda',
        threshold_conf=0.05,
        threshold_multi_persons=0.1,
        input_resolution=(640, 640),
        threshold_iou=0.5,
        threshold_bbox_shape_ratio=0.4,
        cat_id=[1],
        select_type='max',
        strict=True,
        sorted_func=None,
        **kwargs
    ):
        super(Yolo, self).__init__(checkpoint, device=device, **kwargs)

        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 640
        self.input_height = 640

        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func

    def postprocess(self, output, shape_raw, cat_id=[1]):
        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)

        rows = outputs.shape[0]
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]

            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)
        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)
            classid_masks = classid != 3.14159

            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]

            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]

            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)

        results = []
        for i in indices:
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])
        debug_path = os.path.join(os.path.dirname(__file__), "yolo_debug.log")
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"[DEBUG] Threshold={self.threshold_conf}, Scores (first 10)={max_scores[:10]}\n")
        return np.array(results)
        

    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1

            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape

            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1

            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)
            return person_results
        else:
            return None

    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, threshold_conf=None, **kwargs):
        # Apply custom threshold if provided
        if threshold_conf is not None:
            self.threshold_conf = threshold_conf

        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result

    def forward(self, img, shape_raw, **kwargs):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        session = self._live_session()
        outputs = session.run(None, {session.get_inputs()[0].name: img})[0]
        person_results = [
            [{'bbox': np.array([0., 0., 1. * shape_raw[i][1], 1. * shape_raw[i][0], -1]), 'track_id': -1}]
            for i in range(len(outputs))
        ]

        for i in range(len(outputs)):
            # Pass threshold_conf explicitly so runtime updates are applied
            self.postprocess_threading(outputs, shape_raw, person_results, i, threshold_conf=self.threshold_conf, **kwargs)

        return person_results


class ViTPose(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', **kwargs):
        super(ViTPose, self).__init__(checkpoint, device=device)

    def forward(self, img, center, scale, **kwargs):
        session = self._live_session()
        heatmaps = session.run([], {session.get_inputs()[0].name: img})[0]
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=center,
            scale=scale * 200,
            unbiased=True,
            use_udp=False
        )
        return np.concatenate([points, prob], axis=2)
