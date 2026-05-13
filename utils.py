# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import cv2
import math
import random
import numpy as np

def get_mask_boxes(mask):
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    body_bbox = get_mask_boxes(body_mask)

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = np.int32(bbox_wh[0] / w_len)
    h_slice = np.int32(bbox_wh[1] / h_len)

    for each_w in range(body_bbox[0], body_bbox[2], w_slice):
        w_start = min(each_w, body_bbox[2])
        w_end = min((each_w + w_slice), body_bbox[2])
        for each_h in range(body_bbox[1], body_bbox[3], h_slice):
            h_start = min(each_h, body_bbox[3])
            h_end = min((each_h + h_slice), body_bbox[3])
            if body_mask[h_start:h_end, w_start:w_end].sum() > 0:
                body_mask[h_start:h_end, w_start:w_end] = 1

    return body_mask

def get_mask_body_img(img_copy, hand_mask, k=7, iterations=1):
    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(hand_mask, kernel, iterations=iterations)
    mask_hand_img = img_copy * (1 - dilation[:, :, None])

    return mask_hand_img, dilation


def get_face_bboxes(kp2ds, scale, image_shape, ratio_aug):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[23:91, :2]

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)


    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    if ratio_aug:
        if random.random() > 0.5:
            delta_width += random.uniform(0, initial_width // 10)
        else:
            delta_height += random.uniform(0, initial_height // 10)

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):

    target_ratio = orig_w / orig_h

    def check_valid(w, h):

        if w <= 0 or h <= 0:
            return False
        return (w * h <= target_area and
                w % divisor == 0 and
                h % divisor == 0)

    def get_ratio_diff(w, h):

        return abs(w / h - target_ratio)

    def round_to_64(value, round_up=False, divisor=64):

        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    possible_sizes = []

    max_area_h = int(np.sqrt(target_area / target_ratio))
    max_area_w = int(max_area_h * target_ratio)

    max_h = round_to_64(max_area_h, round_up=True, divisor=divisor)
    max_w = round_to_64(max_area_w, round_up=True, divisor=divisor)

    for h in range(divisor, max_h + divisor, divisor):
        ideal_w = h * target_ratio

        w_down = round_to_64(ideal_w)
        w_up = round_to_64(ideal_w, round_up=True)

        for w in [w_down, w_up]:
            if check_valid(w, h, divisor):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    if not possible_sizes:
        raise ValueError("Can not find suitable size")

    possible_sizes.sort(key=lambda x: (-x[0] * x[1], x[2]))

    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def resize_by_area(image, target_area, keep_aspect_ratio=True, divisor=64, padding_color=(0, 0, 0)):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except:
        aspect_ratio = w / h

        if keep_aspect_ratio:
            new_h = math.sqrt(target_area / aspect_ratio)
            new_w = target_area / new_h
        else:
            new_w = new_h = math.sqrt(target_area)

        new_w, new_h = int((new_w // divisor) * divisor), int((new_h // divisor) * divisor)

    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR

    resized_image = padding_resize(image, height=new_h, width=new_w, padding_color=padding_color,
                                    interpolation=interpolation)
    return resized_image


def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel), dtype=img_ori.dtype)
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[:, padding: padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[padding: padding + new_height, :, :] = img

    return img_pad

def resize_to_bounds(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR, extra_padding=64, crop_target_image=None):
    # Find non-black pixel bounds
    if crop_target_image is not None:
        ref = crop_target_image
        if ref.ndim == 2:
            mask = ref > 0
        else:
            mask = np.any(ref != 0, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            # All black, fallback to full image
            y0, x0 = 0, 0
            y1, x1 = img_ori.shape[0], img_ori.shape[1]
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            # Intended crop bounds with padding
            pad_y0 = y0 - extra_padding
            pad_x0 = x0 - extra_padding
            pad_y1 = y1 + extra_padding
            pad_x1 = x1 + extra_padding
            # Actual crop bounds clipped to image
            crop_y0 = max(pad_y0, 0)
            crop_x0 = max(pad_x0, 0)
            crop_y1 = min(pad_y1, img_ori.shape[0])
            crop_x1 = min(pad_x1, img_ori.shape[1])
        crop_img = img_ori[crop_y0:crop_y1, crop_x0:crop_x1]
        # Pad if needed
        pad_top = crop_y0 - pad_y0
        pad_left = crop_x0 - pad_x0
        pad_bottom = pad_y1 - crop_y1
        pad_right = pad_x1 - crop_x1
        if any([pad_top, pad_left, pad_bottom, pad_right]):
            channel = crop_img.shape[2] if crop_img.ndim == 3 else 1
            crop_img = np.pad(
                crop_img,
                ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) if channel > 1 else ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant', constant_values=0
            )
    else:
        if img_ori.ndim == 2:
            mask = img_ori > 0
        else:
            mask = np.any(img_ori != 0, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            # All black, fallback to original
            crop_img = img_ori
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            pad_y0 = y0 - extra_padding
            pad_x0 = x0 - extra_padding
            pad_y1 = y1 + extra_padding
            pad_x1 = x1 + extra_padding
            crop_y0 = max(pad_y0, 0)
            crop_x0 = max(pad_x0, 0)
            crop_y1 = min(pad_y1, img_ori.shape[0])
            crop_x1 = min(pad_x1, img_ori.shape[1])
            crop_img = img_ori[crop_y0:crop_y1, crop_x0:crop_x1]
            pad_top = crop_y0 - pad_y0
            pad_left = crop_x0 - pad_x0
            pad_bottom = pad_y1 - crop_y1
            pad_right = pad_x1 - crop_x1
            if any([pad_top, pad_left, pad_bottom, pad_right]):
                channel = crop_img.shape[2] if crop_img.ndim == 3 else 1
                crop_img = np.pad(
                    crop_img,
                    ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) if channel > 1 else ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0
                )

    ori_height = crop_img.shape[0]
    ori_width = crop_img.shape[1]
    channel = crop_img.shape[2] if crop_img.ndim == 3 else 1

    img_pad = np.zeros((height, width, channel), dtype=crop_img.dtype)
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        for c in range(channel):
            img_pad[:, :, c] = padding_color[c % len(padding_color)]

    # Resize cropped image to fit target size, preserving aspect ratio
    crop_aspect = ori_width / ori_height
    target_aspect = width / height
    if crop_aspect > target_aspect:
        new_width = width
        new_height = int(width / crop_aspect)
    else:
        new_height = height
        new_width = int(height * crop_aspect)
    img = cv2.resize(crop_img, (new_width, new_height), interpolation=interpolation)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    y_pad = (height - new_height) // 2
    x_pad = (width - new_width) // 2
    img_pad[y_pad:y_pad + new_height, x_pad:x_pad + new_width, :] = img

    return img_pad


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):

    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)

    return frame_indices.tolist()


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


# ============================================================
# Wan-Animate paper-driven face preprocessing helpers
# ============================================================
# These helpers implement the four levers identified in the
# Wan-Animate paper (arXiv:2509.14055) for improving gaze
# reenactment fidelity:
#   1. Eye-centred face crop  (place eyes in upper third)
#   2. Stabilised crop position across frames
#   3. CFG on face conditioning (passthrough output, see nodes.py)
#   4. Quality gating (blur / brightness on the eye region)


def adjust_bbox_eye_upper_third(face_bbox, eye_xy_pixel, frame_w, frame_h,
                                 eye_y_fraction=0.30):
    """Vertically shift a face bbox so eye landmarks fall at ``eye_y_fraction``
    of the crop height (measured from the top).

    Args:
        face_bbox:        (x1, x2, y1, y2) in pixel space.
        eye_xy_pixel:     (eye_cx, eye_cy) midpoint of both eye centres in
                          full-frame pixel space.
        frame_w, frame_h: full frame dimensions.
        eye_y_fraction:   target normalised eye row inside the crop
                          (default 0.30 = upper third).

    Returns:
        New (x1, x2, y1, y2) bbox of identical width/height, vertically
        shifted (and clamped to image bounds). Width/height never change.
    """
    x1, x2, y1, y2 = face_bbox
    crop_h = int(y2 - y1)
    if crop_h <= 0 or eye_xy_pixel is None:
        return (int(x1), int(x2), int(y1), int(y2))
    _eye_cx, eye_cy = eye_xy_pixel
    desired_y1 = float(eye_cy) - float(eye_y_fraction) * float(crop_h)
    new_y1 = int(np.clip(desired_y1, 0, max(0, frame_h - crop_h)))
    new_y2 = new_y1 + crop_h
    return (int(x1), int(x2), int(new_y1), int(new_y2))


def compute_eye_midpoint_from_face_kps(face_kps_norm, frame_w, frame_h):
    """Return the pixel-space midpoint of both eyes from a normalised
    dlib-68 face landmark array (the layout used by
    ``pose_metas[i]['keypoints_face']``).

    Right-eye contour: dlib 36-41 -> array indices 37-42.
    Left-eye contour:  dlib 42-47 -> array indices 43-48.
    Index 0 is the body-anchored face anchor (skipped).

    Returns:
        (eye_cx, eye_cy) in pixel space, or ``None`` if landmarks
        are missing/invalid.
    """
    if face_kps_norm is None:
        return None
    arr = np.asarray(face_kps_norm)
    if arr.ndim != 2 or arr.shape[0] < 49 or arr.shape[1] < 2:
        return None
    eye_idx = list(range(37, 49))
    eye_pts = arr[eye_idx, :2]
    if not np.isfinite(eye_pts).all():
        return None
    if arr.shape[1] >= 3:
        conf = arr[eye_idx, 2]
        if float(np.mean(conf)) < 0.05:
            return None
    eye_norm = np.mean(eye_pts, axis=0)
    return (float(eye_norm[0]) * float(frame_w),
            float(eye_norm[1]) * float(frame_h))


def compute_frame_blur_score(frame_rgb):
    """Laplacian variance -- higher = sharper.

    Args:
        frame_rgb: (H, W, 3) array, float32 [0,1] or uint8.
    Returns:
        float Laplacian variance.
    """
    arr = frame_rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_eye_region_brightness(frame_rgb, top_frac=0.30, bottom_frac=0.55):
    """Mean luma of a horizontal strip across the eye region of a face crop.

    With eye-upper-third alignment, eyes fall around 25-35% of the crop
    height. A strip from ``top_frac`` to ``bottom_frac`` covers the eye
    sockets safely.

    Returns:
        Mean luma in [0, 1].
    """
    arr = frame_rgb
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    h = arr.shape[0]
    y0 = int(max(0, top_frac * h))
    y1 = int(min(h, bottom_frac * h))
    if y1 <= y0:
        return 0.0
    strip = arr[y0:y1]
    if strip.ndim == 3:
        luma = 0.2126 * strip[..., 0] + 0.7152 * strip[..., 1] + 0.0722 * strip[..., 2]
    else:
        luma = strip
    return float(np.mean(luma))

