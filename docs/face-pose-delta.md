# Face/Pose Delta Editor

Move individual face points on chosen frames, and let the change follow the face through the clip, even when the head
turns or moves toward the camera.

## Where it goes

```
Pose and Face Detection (V2) ──pose_data──► Face/Pose Delta Editor ──pose_data──► Draw ViT Pose (V2) ──► Wan Animate
```

It works on the face points Pose and Face Detection (V2) finds on every frame. These are the 68 points of the iBUG-68
layout: jaw, brows, nose, eyes and mouth. The edited pose data goes on like the original. Face Controller 3D can sit
before or after it:
- **Face Controller 3D** turns named dials (smile, brow raise, jaw open...), rotates the head and moves the gaze.
- **Face/Pose Delta Editor** moves the exact points you pick, by the amount you set.

The `landmarks` input takes other face landmarks instead, e.g. FantasyPortrait's (MediaPipe FaceMesh, 468 points). When
`pose_data` is connected it wins.

## Why the edit follows the face

An edit is stored relative to the face, not the frame. On every frame the node measures the face:
- its centre;
- its size, the distance between the outer eye corners (iBUG-68 points 36 and 45).

An edit of 0.1 moves a point by a tenth of that distance. On a frame where the face is twice as big, the point moves
twice as many pixels. A pixel offset would stay put while the head moves away from it.

## Making an edit

1. Connect `pose_data`, and run once, so you know the clip's frame numbers.
2. Press **Edit landmark deltas…** on the node.
3. Pick a frame, then a point. The buttons name the common ones (eye corners, brows, mouth corners, lips, nose tip,
   chin); with `pose_data` they use iBUG-68 numbers, with `landmarks` MediaPipe ones.
4. Set `dx` / `dy` in face units (1.0 = eye-corner distance; +x is right, +y is down). Small values go a long way: 0.05
   is a clear change.
5. **Add / replace keyframe**. Repeat for more points or more frames.
6. Choose how edits blend between keyframes (`ease`) and what happens outside them: `hold` keeps the nearest edit,
   `zero` drops the edit (with one keyframe, every other frame is unedited), `loop` repeats the keyframed span. Then
   **Save → node**.

Example: the left mouth corner (iBUG 48) lowered a little on frame 0, more by frame 30:

```json
{
  "keyframes": [
    {"frame": 0,  "deltas": {"48": [0.0, 0.02]}},
    {"frame": 30, "deltas": {"48": [0.0, 0.08]}}
  ],
  "ease": "smooth_step",
  "extrapolate": "hold"
}
```

## Details

- A new node changes nothing until you author an edit.
- Frames where no face was found are passed through unchanged; keyframe numbers are always the clip's frame numbers.
- `left_eye_idx` / `right_eye_idx`: -1 (the default) picks the outer eye corners from the face's point count (36 / 45 for
  iBUG-68, 33 / 263 for MediaPipe). Set them yourself for another layout.
- `anchor_mode` `centroid` measures the face by all its points instead of the eye corners. It is for landmark sets
  without eyes.
- `external_anchors_json` takes per-frame centres and sizes from another tracker.
- `info_json` reports the frames with and without a face, the keyframes used and the blend weights.

Moved here from ComfyUI-CustomNodePacks in 2026-10 (L7.65 P21). The node id is unchanged, so saved workflows load it.
