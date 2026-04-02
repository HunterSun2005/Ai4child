import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

COCO_WHOLEBODY_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [0, 255, 0],      # 5: left_shoulder
    [255, 128, 0],    # 6: right_shoulder
    [0, 255, 0],      # 7: left_elbow
    [255, 128, 0],    # 8: right_elbow
    [0, 255, 0],      # 9: left_wrist
    [255, 128, 0],    # 10: right_wrist
    [0, 255, 0],      # 11: left_hip
    [255, 128, 0],    # 12: right_hip
    [0, 255, 0],      # 13: left_knee
    [255, 128, 0],    # 14: right_knee
    [0, 255, 0],      # 15: left_ankle
    [255, 128, 0],    # 16: right_ankle
    [255, 128, 0],    # 17: left_big_toe
    [255, 128, 0],    # 18: left_small_toe
    [255, 128, 0],    # 19: left_heel
    [255, 128, 0],    # 20: right_big_toe
    [255, 128, 0],    # 21: right_small_toe
    [255, 128, 0],    # 22: right_heel
    [255, 255, 255],  # 23: face-0
    [255, 255, 255],  # 24: face-1
    [255, 255, 255],  # 25: face-2
    [255, 255, 255],  # 26: face-3
    [255, 255, 255],  # 27: face-4
    [255, 255, 255],  # 28: face-5
    [255, 255, 255],  # 29: face-6
    [255, 255, 255],  # 30: face-7
    [255, 255, 255],  # 31: face-8
    [255, 255, 255],  # 32: face-9
    [255, 255, 255],  # 33: face-10
    [255, 255, 255],  # 34: face-11
    [255, 255, 255],  # 35: face-12
    [255, 255, 255],  # 36: face-13
    [255, 255, 255],  # 37: face-14
    [255, 255, 255],  # 38: face-15
    [255, 255, 255],  # 39: face-16
    [255, 255, 255],  # 40: face-17
    [255, 255, 255],  # 41: face-18
    [255, 255, 255],  # 42: face-19
    [255, 255, 255],  # 43: face-20
    [255, 255, 255],  # 44: face-21
    [255, 255, 255],  # 45: face-22
    [255, 255, 255],  # 46: face-23
    [255, 255, 255],  # 47: face-24
    [255, 255, 255],  # 48: face-25
    [255, 255, 255],  # 49: face-26
    [255, 255, 255],  # 50: face-27
    [255, 255, 255],  # 51: face-28
    [255, 255, 255],  # 52: face-29
    [255, 255, 255],  # 53: face-30
    [255, 255, 255],  # 54: face-31
    [255, 255, 255],  # 55: face-32
    [255, 255, 255],  # 56: face-33
    [255, 255, 255],  # 57: face-34
    [255, 255, 255],  # 58: face-35
    [255, 255, 255],  # 59: face-36
    [255, 255, 255],  # 60: face-37
    [255, 255, 255],  # 61: face-38
    [255, 255, 255],  # 62: face-39
    [255, 255, 255],  # 63: face-40
    [255, 255, 255],  # 64: face-41
    [255, 255, 255],  # 65: face-42
    [255, 255, 255],  # 66: face-43
    [255, 255, 255],  # 67: face-44
    [255, 255, 255],  # 68: face-45
    [255, 255, 255],  # 69: face-46
    [255, 255, 255],  # 70: face-47
    [255, 255, 255],  # 71: face-48
    [255, 255, 255],  # 72: face-49
    [255, 255, 255],  # 73: face-50
    [255, 255, 255],  # 74: face-51
    [255, 255, 255],  # 75: face-52
    [255, 255, 255],  # 76: face-53
    [255, 255, 255],  # 77: face-54
    [255, 255, 255],  # 78: face-55
    [255, 255, 255],  # 79: face-56
    [255, 255, 255],  # 80: face-57
    [255, 255, 255],  # 81: face-58
    [255, 255, 255],  # 82: face-59
    [255, 255, 255],  # 83: face-60
    [255, 255, 255],  # 84: face-61
    [255, 255, 255],  # 85: face-62
    [255, 255, 255],  # 86: face-63
    [255, 255, 255],  # 87: face-64
    [255, 255, 255],  # 88: face-65
    [255, 255, 255],  # 89: face-66
    [255, 255, 255],  # 90: face-67
    [255, 255, 255],  # 91: left_hand_root
    [255, 128, 0],    # 92: left_thumb1
    [255, 128, 0],    # 93: left_thumb2
    [255, 128, 0],    # 94: left_thumb3
    [255, 128, 0],    # 95: left_thumb4
    [255, 153, 255],  # 96: left_forefinger1
    [255, 153, 255],  # 97: left_forefinger2
    [255, 153, 255],  # 98: left_forefinger3
    [255, 153, 255],  # 99: left_forefinger4
    [102, 178, 255],  # 100: left_middle_finger1
    [102, 178, 255],  # 101: left_middle_finger2
    [102, 178, 255],  # 102: left_middle_finger3
    [102, 178, 255],  # 103: left_middle_finger4
    [255, 51, 51],    # 104: left_ring_finger1
    [255, 51, 51],    # 105: left_ring_finger2
    [255, 51, 51],    # 106: left_ring_finger3
    [255, 51, 51],    # 107: left_ring_finger4
    [0, 255, 0],      # 108: left_pinky_finger1
    [0, 255, 0],      # 109: left_pinky_finger2
    [0, 255, 0],      # 110: left_pinky_finger3
    [0, 255, 0],      # 111: left_pinky_finger4
    [255, 255, 255],  # 112: right_hand_root
    [255, 128, 0],    # 113: right_thumb1
    [255, 128, 0],    # 114: right_thumb2
    [255, 128, 0],    # 115: right_thumb3
    [255, 128, 0],    # 116: right_thumb4
    [255, 153, 255],  # 117: right_forefinger1
    [255, 153, 255],  # 118: right_forefinger2
    [255, 153, 255],  # 119: right_forefinger3
    [255, 153, 255],  # 120: right_forefinger4
    [102, 178, 255],  # 121: right_middle_finger1
    [102, 178, 255],  # 122: right_middle_finger2
    [102, 178, 255],  # 123: right_middle_finger3
    [102, 178, 255],  # 124: right_middle_finger4
    [255, 51, 51],    # 125: right_ring_finger1
    [255, 51, 51],    # 126: right_ring_finger2
    [255, 51, 51],    # 127: right_ring_finger3
    [255, 51, 51],    # 128: right_ring_finger4
    [0, 255, 0],      # 129: right_pinky_finger1
    [0, 255, 0],      # 130: right_pinky_finger2
    [0, 255, 0],      # 131: right_pinky_finger3
    [0, 255, 0],      # 132: right_pinky_finger4
]

COCO_WHOLEBODY_SKELETON_INFO = {
    0: dict(link=(15, 13), id=0, color=[0, 255, 0]),
    1: dict(link=(13, 11), id=1, color=[0, 255, 0]),
    2: dict(link=(16, 14), id=2, color=[255, 128, 0]),
    3: dict(link=(14, 12), id=3, color=[255, 128, 0]),
    4: dict(link=(11, 12), id=4, color=[51, 153, 255]),
    5: dict(link=(5, 11), id=5, color=[51, 153, 255]),
    6: dict(link=(6, 12), id=6, color=[51, 153, 255]),
    7: dict(link=(5, 6), id=7, color=[51, 153, 255]),
    8: dict(link=(5, 7), id=8, color=[0, 255, 0]),
    9: dict(link=(6, 8), id=9, color=[255, 128, 0]),
    10: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    11: dict(link=(8, 10), id=11, color=[255, 128, 0]),
    12: dict(link=(1, 2), id=12, color=[51, 153, 255]),
    13: dict(link=(0, 1), id=13, color=[51, 153, 255]),
    14: dict(link=(0, 2), id=14, color=[51, 153, 255]),
    15: dict(link=(1, 3), id=15, color=[51, 153, 255]),
    16: dict(link=(2, 4), id=16, color=[51, 153, 255]),
    17: dict(link=(3, 5), id=17, color=[51, 153, 255]),
    18: dict(link=(4, 6), id=18, color=[51, 153, 255]),
    19: dict(link=(15, 17), id=19, color=[0, 255, 0]),
    20: dict(link=(15, 18), id=20, color=[0, 255, 0]),
    21: dict(link=(15, 19), id=21, color=[0, 255, 0]),
    22: dict(link=(16, 20), id=22, color=[255, 128, 0]),
    23: dict(link=(16, 21), id=23, color=[255, 128, 0]),
    24: dict(link=(16, 22), id=24, color=[255, 128, 0]),
    25: dict(link=(91, 92), id=25, color=[255, 128, 0]),
    26: dict(link=(92, 93), id=26, color=[255, 128, 0]),
    27: dict(link=(93, 94), id=27, color=[255, 128, 0]),
    28: dict(link=(94, 95), id=28, color=[255, 128, 0]),
    29: dict(link=(91, 96), id=29, color=[255, 153, 255]),
    30: dict(link=(96, 97), id=30, color=[255, 153, 255]),
    31: dict(link=(97, 98), id=31, color=[255, 153, 255]),
    32: dict(link=(98, 99), id=32, color=[255, 153, 255]),
    33: dict(link=(91, 100), id=33, color=[102, 178, 255]),
    34: dict(link=(100, 101), id=34, color=[102, 178, 255]),
    35: dict(link=(101, 102), id=35, color=[102, 178, 255]),
    36: dict(link=(102, 103), id=36, color=[102, 178, 255]),
    37: dict(link=(91, 104), id=37, color=[255, 51, 51]),
    38: dict(link=(104, 105), id=38, color=[255, 51, 51]),
    39: dict(link=(105, 106), id=39, color=[255, 51, 51]),
    40: dict(link=(106, 107), id=40, color=[255, 51, 51]),
    41: dict(link=(91, 108), id=41, color=[0, 255, 0]),
    42: dict(link=(108, 109), id=42, color=[0, 255, 0]),
    43: dict(link=(109, 110), id=43, color=[0, 255, 0]),
    44: dict(link=(110, 111), id=44, color=[0, 255, 0]),
    45: dict(link=(112, 113), id=45, color=[255, 128, 0]),
    46: dict(link=(113, 114), id=46, color=[255, 128, 0]),
    47: dict(link=(114, 115), id=47, color=[255, 128, 0]),
    48: dict(link=(115, 116), id=48, color=[255, 128, 0]),
    49: dict(link=(112, 117), id=49, color=[255, 153, 255]),
    50: dict(link=(117, 118), id=50, color=[255, 153, 255]),
    51: dict(link=(118, 119), id=51, color=[255, 153, 255]),
    52: dict(link=(119, 120), id=52, color=[255, 153, 255]),
    53: dict(link=(112, 121), id=53, color=[102, 178, 255]),
    54: dict(link=(121, 122), id=54, color=[102, 178, 255]),
    55: dict(link=(122, 123), id=55, color=[102, 178, 255]),
    56: dict(link=(123, 124), id=56, color=[102, 178, 255]),
    57: dict(link=(112, 125), id=57, color=[255, 51, 51]),
    58: dict(link=(125, 126), id=58, color=[255, 51, 51]),
    59: dict(link=(126, 127), id=59, color=[255, 51, 51]),
    60: dict(link=(127, 128), id=60, color=[255, 51, 51]),
    61: dict(link=(112, 129), id=61, color=[0, 255, 0]),
    62: dict(link=(129, 130), id=62, color=[0, 255, 0]),
    63: dict(link=(130, 131), id=63, color=[0, 255, 0]),
    64: dict(link=(131, 132), id=64, color=[0, 255, 0]),
}

def _to_bgr(color_rgb: List[int]) -> Tuple[int, int, int]:
    return int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0])

def _clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    x_i = int(round(max(0, min(w - 1, x))))
    y_i = int(round(max(0, min(h - 1, y))))
    return x_i, y_i

def draw_one_instance(
    canvas: np.ndarray,
    instance: Dict,
    score_thr: float,
    kp_radius: int,
    line_thickness: int,
) -> None:
    h, w = canvas.shape[:2]
    keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] == 0 or keypoints.shape[1] < 2:
        return

    scores = instance.get("keypoint_scores", None)
    if scores is None:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != keypoints.shape[0]:
            scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    # Draw bbox (xywh).
    bbox = instance.get("gt_bbox_xywh_px", None)
    if bbox is not None and len(bbox) >= 4:
        x, y, bw, bh = bbox[:4]
        x1, y1 = _clamp_point(float(x), float(y), w, h)
        x2, y2 = _clamp_point(float(x + bw), float(y + bh), w, h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        if "obj_id" in instance:
            cv2.putText(
                canvas,
                f"id={instance['obj_id']}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Draw skeleton links.
    for _, info in COCO_WHOLEBODY_SKELETON_INFO.items():
        i, j = info["link"]
        if i >= keypoints.shape[0] or j >= keypoints.shape[0]:
            continue
        if scores[i] < score_thr or scores[j] < score_thr:
            continue
        xi, yi = _clamp_point(float(keypoints[i, 0]), float(keypoints[i, 1]), w, h)
        xj, yj = _clamp_point(float(keypoints[j, 0]), float(keypoints[j, 1]), w, h)
        cv2.line(canvas, (xi, yi), (xj, yj), _to_bgr(info["color"]), line_thickness, cv2.LINE_AA)

    # Draw keypoints on top.
    for idx in range(keypoints.shape[0]):
        if scores[idx] < score_thr:
            continue
        xk, yk = _clamp_point(float(keypoints[idx, 0]), float(keypoints[idx, 1]), w, h)
        color_rgb = COCO_WHOLEBODY_KPTS_COLORS[idx] if idx < len(COCO_WHOLEBODY_KPTS_COLORS) else [255, 255, 255]
        cv2.circle(canvas, (xk, yk), kp_radius, _to_bgr(color_rgb), -1, cv2.LINE_AA)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pose keypoints + bbox from per-frame JSON files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing frame_*.json.")
    parser.add_argument("--output_dir", type=str, default="vis_output", help="Output folder.")
    parser.add_argument(
        "--mode",
        choices=["images", "video", "both"],
        default="both",
        help="Export images, video, or both.",
    )
    parser.add_argument("--score_thr", type=float, default=0.2, help="Keypoint score threshold.")
    parser.add_argument("--kp_radius", type=int, default=2, help="Keypoint circle radius.")
    parser.add_argument("--line_thickness", type=int, default=2, help="Skeleton line thickness.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Only render first N frames if > 0.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("frame_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No frame_*.json found in: {input_dir}")

    if args.max_frames > 0:
        json_files = json_files[: args.max_frames]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "frames_vis"
    if args.mode in ("images", "both"):
        images_dir.mkdir(parents=True, exist_ok=True)

    first = read_json(json_files[0])
    vi = first.get("video_info", {})
    width = int(vi.get("width", 1920))
    height = int(vi.get("height", 1080))
    fps = float(vi.get("fps", 30))
    video_name = vi.get("video_name", "pose_vis.mp4")
    video_path = output_dir / f"{Path(video_name).stem}_pose_vis.mp4"

    writer = None
    if args.mode in ("video", "both"):
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for: {video_path}")

    total = len(json_files)
    for idx, js in enumerate(json_files, start=1):
        data = read_json(js)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for inst in data.get("instance_info", []):
            draw_one_instance(
                canvas=canvas,
                instance=inst,
                score_thr=args.score_thr,
                kp_radius=args.kp_radius,
                line_thickness=args.line_thickness,
            )

        # Frame label.
        frame_index = data.get("frame_index", idx - 1)
        cv2.putText(
            canvas,
            f"frame={frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        if args.mode in ("images", "both"):
            out_img = images_dir / f"{js.stem}_vis.jpg"
            cv2.imwrite(str(out_img), canvas)
        if writer is not None:
            writer.write(canvas)

        if idx % 50 == 0 or idx == total:
            print(f"[{idx}/{total}] rendered")

    if writer is not None:
        writer.release()
        print(f"Video saved to: {video_path}")
    if args.mode in ("images", "both"):
        print(f"Frames saved to: {images_dir}")


if __name__ == "__main__":
    main()
