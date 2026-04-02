import csv
import glob
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allows preprocess-only usage without torch installed
    torch = None

    class Dataset:  # type: ignore
        pass

from .constants import DIRECTION_TO_INDEX, TRACK2_LABEL_TO_INDEX
from .graph import AichildGraph


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_track1_labels(path: str) -> Dict[int, Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    labels = {}
    for row in rows:
        sid = int(row["patient_id"])
        left = [int(row["left"][str(i)]) for i in range(1, 18)]
        right = [int(row["right"][str(i)]) for i in range(1, 18)]
        labels[sid] = {"left": left, "right": right}
    return labels


def _load_track2_labels(path: str) -> Dict[int, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    labels = {}
    for row in rows:
        sid = int(row["patient_id"])
        left = TRACK2_LABEL_TO_INDEX[row["left"]["gait_subtype"]]
        right = TRACK2_LABEL_TO_INDEX[row["right"]["gait_subtype"]]
        labels[sid] = {"left": left, "right": right}
    return labels


def _detect_direction(clip_name: str) -> Optional[str]:
    for key in DIRECTION_TO_INDEX.keys():
        if f"_{key}_" in clip_name:
            return key
    return None


def _load_frame_keypoints(
    frame_path: str,
    keypoint_indices: List[int],
    score_thr: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    with open(frame_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inst_list = data.get("instance_info", [])
    if not inst_list:
        return None, None

    instance = inst_list[0]
    keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] <= max(keypoint_indices):
        return None, None

    scores = np.asarray(
        instance.get("keypoint_scores", [1.0] * keypoints.shape[0]), dtype=np.float32
    )
    if scores.shape[0] != keypoints.shape[0]:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    keypoints = keypoints[keypoint_indices]  # (V, 2)
    scores = scores[keypoint_indices]        # (V,)

    low = scores < score_thr
    if np.any(low):
        keypoints = keypoints.copy()
        keypoints[low] = np.nan

    return keypoints, scores


def _fill_nan_forward_backward(values: np.ndarray) -> np.ndarray:
    """
    Fill NaNs along time axis.
    values: (T, V, C)
    """
    out = values.copy()
    T = out.shape[0]

    # Forward fill.
    for t in range(1, T):
        mask = np.isnan(out[t])
        out[t][mask] = out[t - 1][mask]

    # Backward fill for leading NaNs.
    for t in range(T - 2, -1, -1):
        mask = np.isnan(out[t])
        out[t][mask] = out[t + 1][mask]

    out[np.isnan(out)] = 0.0
    return out


def _resample_time(values: np.ndarray, target_t: int) -> np.ndarray:
    """
    values: (T, V, C)
    return: (target_t, V, C)
    """
    src_t = values.shape[0]
    if src_t == target_t:
        return values

    if src_t == 1:
        return np.repeat(values, target_t, axis=0)

    src_idx = np.arange(src_t, dtype=np.float32)
    tgt_idx = np.linspace(0, src_t - 1, target_t, dtype=np.float32)

    tgt = np.zeros((target_t, values.shape[1], values.shape[2]), dtype=np.float32)
    for v in range(values.shape[1]):
        for c in range(values.shape[2]):
            tgt[:, v, c] = np.interp(tgt_idx, src_idx, values[:, v, c])
    return tgt


def _normalize_xy(values: np.ndarray) -> np.ndarray:
    """
    values: (T, V, 2)
    - center by pelvis midpoint (left/right hip: 11/12)
    - normalize by robust body scale
    """
    out = values.copy()

    center = (out[:, 11, :] + out[:, 12, :]) * 0.5
    out = out - center[:, None, :]

    shoulder = np.linalg.norm(out[:, 5, :] - out[:, 6, :], axis=1)
    hip = np.linalg.norm(out[:, 11, :] - out[:, 12, :], axis=1)
    valid = np.concatenate([shoulder[shoulder > 1e-6], hip[hip > 1e-6]], axis=0)
    scale = float(np.median(valid)) if valid.size > 0 else 1.0
    out /= max(scale, 1e-6)
    return out


def _clip_to_joint_tensor(xy: np.ndarray) -> np.ndarray:
    """
    xy: (T, V, 2)
    returns: (C=2, T, V, M=1)
    """
    T, V, _ = xy.shape
    joint = np.zeros((2, T, V, 1), dtype=np.float32)
    joint[0, :, :, 0] = xy[:, :, 0]
    joint[1, :, :, 0] = xy[:, :, 1]
    return joint


def preprocess_dataset(config: dict, max_clips: int = -1, overwrite: bool = False) -> dict:
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    comp_cfg = config["competition"]

    dataset_root = os.path.abspath(paths_cfg["dataset_root"])
    cache_dir = os.path.abspath(paths_cfg["cache_dir"])
    manifest_path = os.path.abspath(paths_cfg["manifest_path"])

    track1_labels = _load_track1_labels(os.path.abspath(paths_cfg["track1_label"]))
    track2_labels = _load_track2_labels(os.path.abspath(paths_cfg["track2_label"]))
    track1_test_ids = set(int(x) for x in comp_cfg.get("track1_test_ids", []))
    track2_test_ids = set(int(x) for x in comp_cfg["track2_test_ids"])

    keypoint_indices = data_cfg["keypoint_indices"]
    target_t = int(data_cfg["num_frame"])
    score_thr = float(data_cfg["score_thr"])

    _ensure_dir(cache_dir)
    _ensure_dir(os.path.dirname(manifest_path))

    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(d)
    )

    rows: List[Dict[str, str]] = []
    total = 0
    dropped = 0

    for subject_dir in subject_dirs:
        sid_name = os.path.basename(subject_dir)
        if not sid_name.isdigit():
            continue
        sid = int(sid_name)

        clip_dirs = sorted(glob.glob(os.path.join(subject_dir, "*_filtered_pose")))
        for clip_dir in clip_dirs:
            if max_clips > 0 and total >= max_clips:
                break
            total += 1

            clip_name = os.path.basename(clip_dir)
            direction = _detect_direction(clip_name)
            if direction is None:
                dropped += 1
                continue

            frame_files = sorted(glob.glob(os.path.join(clip_dir, "frame_*.json")))
            if not frame_files:
                dropped += 1
                continue

            cache_path = os.path.join(cache_dir, f"{clip_name}.npz")
            if not overwrite and os.path.exists(cache_path):
                num_frames = int(np.load(cache_path)["joint"].shape[1])
            else:
                xy_list = []
                score_list = []
                for frame_path in frame_files:
                    xy, sc = _load_frame_keypoints(frame_path, keypoint_indices, score_thr)
                    if xy is None:
                        xy_list.append(np.full((len(keypoint_indices), 2), np.nan, dtype=np.float32))
                        score_list.append(np.zeros((len(keypoint_indices),), dtype=np.float32))
                    else:
                        xy_list.append(xy)
                        score_list.append(sc)

                xy_seq = np.asarray(xy_list, dtype=np.float32)        # (T_raw, V, 2)
                sc_seq = np.asarray(score_list, dtype=np.float32)      # (T_raw, V)

                if np.isnan(xy_seq).all():
                    dropped += 1
                    continue

                xy_seq = _fill_nan_forward_backward(xy_seq)
                sc_seq = _fill_nan_forward_backward(sc_seq[:, :, None])[:, :, 0]

                xy_seq = _resample_time(xy_seq, target_t)
                sc_seq = _resample_time(sc_seq[:, :, None], target_t)[:, :, 0]
                xy_seq = _normalize_xy(xy_seq)

                joint = _clip_to_joint_tensor(xy_seq)
                score = sc_seq[None, :, :, None].astype(np.float32)

                np.savez_compressed(
                    cache_path,
                    joint=joint,
                    score=score,
                    source_clip=clip_dir,
                )
                num_frames = int(joint.shape[1])

            t1 = track1_labels.get(sid)
            t2 = track2_labels.get(sid)

            row = {
                "subject_id": str(sid),
                "clip_id": clip_name,
                "clip_dir": clip_dir,
                "cache_path": cache_path,
                "direction": direction,
                "direction_idx": str(DIRECTION_TO_INDEX[direction]),
                "num_frames": str(num_frames),
                "is_track1_test": "1" if sid in track1_test_ids else "0",
                "is_track2_test": "1" if sid in track2_test_ids else "0",
                "has_track1_label": "1" if t1 is not None else "0",
                "has_track2_label": "1" if t2 is not None else "0",
                "track1_left": "" if t1 is None else ";".join(str(x) for x in t1["left"]),
                "track1_right": "" if t1 is None else ";".join(str(x) for x in t1["right"]),
                "track2_left": "" if t2 is None else str(t2["left"]),
                "track2_right": "" if t2 is None else str(t2["right"]),
            }
            rows.append(row)

        if max_clips > 0 and total >= max_clips:
            break

    fieldnames = [
        "subject_id",
        "clip_id",
        "clip_dir",
        "cache_path",
        "direction",
        "direction_idx",
        "num_frames",
        "is_track1_test",
        "is_track2_test",
        "has_track1_label",
        "has_track2_label",
        "track1_left",
        "track1_right",
        "track2_left",
        "track2_right",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    info = {
        "manifest_path": manifest_path,
        "total_clips_scanned": total,
        "total_rows": len(rows),
        "dropped_clips": dropped,
    }
    logging.info("Preprocess done: %s", info)
    return info


def load_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {
                "subject_id": int(row["subject_id"]),
                "clip_id": row["clip_id"],
                "clip_dir": row["clip_dir"],
                "cache_path": row["cache_path"],
                "direction": row["direction"],
                "direction_idx": int(row["direction_idx"]),
                "num_frames": int(row["num_frames"]),
                "is_track1_test": row.get("is_track1_test", "0") == "1",
                "is_track2_test": row.get("is_track2_test", "0") == "1",
                "has_track1_label": row["has_track1_label"] == "1",
                "has_track2_label": row["has_track2_label"] == "1",
                "track1_left": None,
                "track1_right": None,
                "track2_left": None,
                "track2_right": None,
            }
            if parsed["has_track1_label"]:
                parsed["track1_left"] = [int(x) for x in row["track1_left"].split(";")]
                parsed["track1_right"] = [int(x) for x in row["track1_right"].split(";")]
            if parsed["has_track2_label"]:
                parsed["track2_left"] = int(row["track2_left"])
                parsed["track2_right"] = int(row["track2_right"])
            rows.append(parsed)
    return rows


def _resample_input_channels(x: np.ndarray, target_t: int) -> np.ndarray:
    """x: (C, T, V, M)"""
    c, t, v, m = x.shape
    if t == target_t:
        return x
    if t == 1:
        return np.repeat(x, target_t, axis=1)

    src_idx = np.arange(t, dtype=np.float32)
    tgt_idx = np.linspace(0, t - 1, target_t, dtype=np.float32)
    out = np.zeros((c, target_t, v, m), dtype=np.float32)
    for ci in range(c):
        for vi in range(v):
            for mi in range(m):
                out[ci, :, vi, mi] = np.interp(tgt_idx, src_idx, x[ci, :, vi, mi])
    return out


@dataclass
class DatasetOptions:
    inputs: str
    root_joint: int
    num_frame: int
    train: bool
    return_ssl: bool
    jitter_std: float
    temporal_crop_min: float


class AichildClipDataset(Dataset):
    def __init__(self, rows: List[dict], graph: AichildGraph, options: DatasetOptions):
        self.rows = rows
        self.graph = graph
        self.options = options

    def __len__(self):
        return len(self.rows)

    def _augment_joint(self, joint: np.ndarray) -> np.ndarray:
        out = joint.copy()
        _, T, _, _ = out.shape

        # Random temporal crop + resize.
        min_ratio = self.options.temporal_crop_min
        if min_ratio < 1.0:
            crop = int(random.uniform(min_ratio, 1.0) * T)
            crop = max(8, min(crop, T))
            start = random.randint(0, T - crop)
            out = out[:, start : start + crop]
            out = _resample_input_channels(out, T)

        # Mild temporal shift.
        shift = random.randint(-8, 8)
        out = np.roll(out, shift=shift, axis=1)

        # Jitter on coordinates.
        out[:2] += np.random.normal(0.0, self.options.jitter_std, size=out[:2].shape).astype(np.float32)

        # Random scale.
        scale = random.uniform(0.95, 1.05)
        out[:2] *= scale
        return out

    def _build_multi_input(self, joint: np.ndarray) -> np.ndarray:
        """
        joint: (C, T, V, M), C should be 2.
        return: (I, C*2, T, V, M)
        """
        C, T, V, M = joint.shape
        assert C == 2, f"Expected C=2, got {C}"

        joint_f = np.zeros((C * 2, T, V, M), dtype=np.float32)
        velocity_f = np.zeros((C * 2, T, V, M), dtype=np.float32)
        bone_f = np.zeros((C * 2, T, V, M), dtype=np.float32)

        joint_f[:C] = joint
        root = joint[:, :, self.options.root_joint : self.options.root_joint + 1, :]
        joint_f[C:] = joint - root

        velocity_f[:C, :-1] = joint[:, 1:] - joint[:, :-1]
        velocity_f[C:, :-2] = joint[:, 2:] - joint[:, :-2]

        conn = self.graph.connect_joint
        for v in range(V):
            bone_f[:C, :, v, :] = joint[:, :, v, :] - joint[:, :, conn[v], :]

        bone_len = np.sqrt(np.maximum(np.sum(np.square(bone_f[:C]), axis=0), 1e-8))
        for c in range(C):
            bone_f[C + c] = np.arccos(np.clip(bone_f[c] / bone_len, -1.0, 1.0))

        streams = []
        if "J" in self.options.inputs:
            streams.append(joint_f)
        if "V" in self.options.inputs:
            streams.append(velocity_f)
        if "B" in self.options.inputs:
            streams.append(bone_f)

        return np.stack(streams, axis=0)

    def __getitem__(self, index: int):
        if torch is None:
            raise ImportError("PyTorch is required to use AichildClipDataset.")

        row = self.rows[index]
        data = np.load(row["cache_path"])
        joint = data["joint"].astype(np.float32)  # (2, T, V, 1)

        if self.options.train:
            joint_main = self._augment_joint(joint)
            joint_ssl = self._augment_joint(joint)
        else:
            joint_main = joint
            joint_ssl = joint

        x = self._build_multi_input(joint_main)
        x_ssl = self._build_multi_input(joint_ssl)

        direction_idx = int(row["direction_idx"])
        direction = np.zeros((4,), dtype=np.float32)
        direction[direction_idx] = 1.0

        t1_mask = 1.0 if row["has_track1_label"] else 0.0
        t2_mask = 1.0 if row["has_track2_label"] else 0.0

        t1_left = np.zeros((17,), dtype=np.float32)
        t1_right = np.zeros((17,), dtype=np.float32)
        if row["has_track1_label"]:
            t1_left[:] = np.asarray(row["track1_left"], dtype=np.float32)
            t1_right[:] = np.asarray(row["track1_right"], dtype=np.float32)

        t2_left = int(row["track2_left"]) if row["has_track2_label"] else 0
        t2_right = int(row["track2_right"]) if row["has_track2_label"] else 0

        sample = {
            "x": torch.from_numpy(x),
            "x_ssl": torch.from_numpy(x_ssl),
            "direction": torch.from_numpy(direction),
            "track1_left": torch.from_numpy(t1_left),
            "track1_right": torch.from_numpy(t1_right),
            "track1_mask": torch.tensor(t1_mask, dtype=torch.float32),
            "track2_left": torch.tensor(t2_left, dtype=torch.long),
            "track2_right": torch.tensor(t2_right, dtype=torch.long),
            "track2_mask": torch.tensor(t2_mask, dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "clip_id": row["clip_id"],
        }

        return sample
