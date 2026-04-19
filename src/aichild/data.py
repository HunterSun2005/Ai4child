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

_COCO_WHOLEBODY_LR_PAIRS = {
    1: 2, 2: 1,
    3: 4, 4: 3,
    5: 6, 6: 5,
    7: 8, 8: 7,
    9: 10, 10: 9,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
    17: 20, 20: 17,
    18: 21, 21: 18,
    19: 22, 22: 19,
}
for k in range(21):
    _COCO_WHOLEBODY_LR_PAIRS[91 + k] = 112 + k
    _COCO_WHOLEBODY_LR_PAIRS[112 + k] = 91 + k


def _build_lr_flip_index(keypoint_indices: List[int]) -> np.ndarray:
    local_of = {orig: i for i, orig in enumerate(keypoint_indices)}
    flip_index = np.arange(len(keypoint_indices), dtype=np.int64)
    for orig_idx, local_idx in local_of.items():
        pair_orig = _COCO_WHOLEBODY_LR_PAIRS.get(orig_idx, orig_idx)
        pair_local = local_of.get(pair_orig, local_idx)
        flip_index[local_idx] = pair_local
    return flip_index


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


def _resolve_pca_model_path(config: dict) -> str:
    paths_cfg = config["paths"]
    p = paths_cfg.get("pca_model_path", "")
    if p:
        return os.path.abspath(p)
    work_dir = os.path.abspath(paths_cfg["work_dir"])
    return os.path.join(work_dir, "pca_joint_model.npz")


def _fit_axis_pca(samples: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    samples: (N, V)
    returns:
      mean: (V,)
      components: (K, V)
      explained_ratio: (K,)
    """
    if samples.ndim != 2:
        raise ValueError(f"PCA samples must be 2D, got shape={samples.shape}")

    n, v = samples.shape
    if n < 2:
        raise ValueError("PCA requires at least 2 samples.")

    k = min(int(n_components), v)
    mean = samples.mean(axis=0).astype(np.float32)
    centered = (samples - mean).astype(np.float64)

    cov = centered.T @ centered
    cov /= max(n - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    components = eigvecs[:, :k].T.astype(np.float32)  # (K, V)
    total = float(np.maximum(eigvals.sum(), 1e-12))
    explained = (eigvals[:k] / total).astype(np.float32)
    return mean, components, explained


def _fit_and_save_joint_pca(config: dict, rows: List[dict], overwrite: bool = False) -> Optional[str]:
    data_cfg = config["data"]
    pca_cfg = data_cfg.get("pca", {})
    if not bool(pca_cfg.get("enabled", False)):
        return None

    n_components = int(pca_cfg.get("n_components", 0))
    if n_components <= 0:
        raise ValueError("data.pca.n_components must be > 0 when PCA is enabled.")

    pca_path = _resolve_pca_model_path(config)
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    if os.path.exists(pca_path) and not overwrite:
        logging.info("PCA model exists, reuse: %s", pca_path)
        return pca_path

    fit_on = str(pca_cfg.get("fit_on", "non_test")).strip().lower()
    to_bool = lambda x: str(x).strip() in {"1", "true", "True"}
    if fit_on == "all":
        fit_rows = rows
    elif fit_on == "labeled":
        fit_rows = [r for r in rows if to_bool(r["has_track1_label"]) or to_bool(r["has_track2_label"])]
    else:
        # default: non_test
        fit_rows = [r for r in rows if (not to_bool(r["is_track1_test"])) and (not to_bool(r["is_track2_test"]))]

    if not fit_rows:
        logging.warning("No rows matched fit_on=%s for PCA; fallback to all rows.", fit_on)
        fit_rows = rows
    if not fit_rows:
        raise ValueError(f"No rows available for PCA fit (fit_on={fit_on}).")

    frames_per_clip = int(pca_cfg.get("frames_per_clip", 32))
    frames_per_clip = max(frames_per_clip, 1)

    x_list = []
    y_list = []
    for row in fit_rows:
        cache_path = row["cache_path"]
        if not os.path.exists(cache_path):
            continue
        joint = np.load(cache_path)["joint"].astype(np.float32)  # (2, T, V, 1)
        t = joint.shape[1]
        if t <= frames_per_clip:
            idx = np.arange(t, dtype=np.int64)
        else:
            idx = np.linspace(0, t - 1, frames_per_clip).round().astype(np.int64)

        x_list.append(joint[0, idx, :, 0])
        y_list.append(joint[1, idx, :, 0])

    if not x_list or not y_list:
        raise ValueError("No clip cache available to fit PCA model.")

    x_samples = np.concatenate(x_list, axis=0)  # (N, V)
    y_samples = np.concatenate(y_list, axis=0)  # (N, V)
    v = int(x_samples.shape[1])
    if n_components > v:
        raise ValueError(
            f"data.pca.n_components={n_components} cannot exceed keypoint dim V={v}."
        )

    mean_x, comp_x, explained_x = _fit_axis_pca(x_samples, n_components=n_components)
    mean_y, comp_y, explained_y = _fit_axis_pca(y_samples, n_components=n_components)

    np.savez_compressed(
        pca_path,
        mean_x=mean_x,
        components_x=comp_x,
        explained_ratio_x=explained_x,
        mean_y=mean_y,
        components_y=comp_y,
        explained_ratio_y=explained_y,
        keypoint_dim=np.array([v], dtype=np.int64),
        n_components=np.array([int(comp_x.shape[0])], dtype=np.int64),
        fit_on=np.array([fit_on]),
    )
    logging.info(
        "Saved PCA model: %s | dim=%d -> %d | fit_on=%s | clips=%d | samples=%d",
        pca_path,
        v,
        int(comp_x.shape[0]),
        fit_on,
        len(fit_rows),
        int(x_samples.shape[0]),
    )
    logging.info(
        "PCA explained variance | x=%.4f | y=%.4f",
        float(explained_x.sum()),
        float(explained_y.sum()),
    )
    return pca_path


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

    pca_path = _fit_and_save_joint_pca(config, rows, overwrite=overwrite)

    info = {
        "manifest_path": manifest_path,
        "total_clips_scanned": total,
        "total_rows": len(rows),
        "dropped_clips": dropped,
        "pca_model_path": pca_path or "",
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
    score_thr: float
    train: bool
    return_ssl: bool
    jitter_std: float
    temporal_crop_min: float
    use_pca: bool = False
    pca_model_path: str = ""
    use_score: bool = False
    score_clip_min: float = 0.0
    score_clip_max: float = 1.0
    score_power: float = 1.0
    score_only_above_thr: bool = True
    # Temporal augmentation.
    enable_time_aug: bool = True
    temporal_shift_max: int = 8
    speed_min: float = 0.9
    speed_max: float = 1.1
    # Spatial augmentation.
    enable_spatial_aug: bool = True
    rotate_deg: float = 6.0
    translate_std: float = 0.01
    scale_min: float = 0.95
    scale_max: float = 1.05
    # Confidence augmentation.
    enable_conf_aug: bool = True
    conf_noise_std: float = 0.02
    conf_drop_prob: float = 0.05
    conf_low_boost: float = 0.20
    # Left-right flip augmentation.
    enable_lr_flip: bool = False
    flip_prob: float = 0.0
    # Rare-class augmentation (track2).
    enable_rare_aug: bool = False
    rare_track2_indices: Tuple[int, ...] = (3, 4)
    rare_strength: float = 1.5


class AichildClipDataset(Dataset):
    def __init__(self, rows: List[dict], graph: AichildGraph, options: DatasetOptions):
        self.rows = rows
        self.graph = graph
        self.options = options
        self.pca_model = None
        self.lr_flip_index = _build_lr_flip_index(graph.keypoint_indices)
        if self.options.use_pca:
            self.pca_model = self._load_pca_model(self.options.pca_model_path)

    def _load_pca_model(self, path: str) -> dict:
        p = os.path.abspath(path) if path else ""
        if not p or not os.path.exists(p):
            raise FileNotFoundError(
                "PCA is enabled but model file not found. "
                f"Expected: {p if p else '<empty path>'}. "
                "Please run preprocess first or disable data.pca.enabled."
            )
        data = np.load(p)
        model = {
            "mean_x": data["mean_x"].astype(np.float32),
            "components_x": data["components_x"].astype(np.float32),
            "mean_y": data["mean_y"].astype(np.float32),
            "components_y": data["components_y"].astype(np.float32),
            "keypoint_dim": int(data["keypoint_dim"][0]),
            "n_components": int(data["n_components"][0]),
        }
        return model

    def _apply_joint_pca(self, joint: np.ndarray) -> np.ndarray:
        """
        PCA reduce+reconstruct on each frame axis-wise (x/y), keeping original joint size.
        joint: (2, T, V, 1)
        """
        if self.pca_model is None:
            return joint

        out = joint.copy()
        _, t, v, _ = out.shape
        expected_v = int(self.pca_model["keypoint_dim"])
        if v != expected_v:
            raise ValueError(
                f"PCA keypoint dim mismatch: joint V={v}, PCA expects V={expected_v}."
            )

        for axis, mean_key, comp_key in [
            (0, "mean_x", "components_x"),
            (1, "mean_y", "components_y"),
        ]:
            x = out[axis, :, :, 0]  # (T, V)
            mean = self.pca_model[mean_key][None, :]  # (1, V)
            comp = self.pca_model[comp_key]  # (K, V)
            z = (x - mean) @ comp.T         # (T, K)
            x_rec = z @ comp + mean         # (T, V)
            out[axis, :, :, 0] = x_rec.astype(np.float32)
        return out

    def __len__(self):
        return len(self.rows)

    def _is_rare_track2_row(self, row: dict) -> bool:
        if not self.options.enable_rare_aug:
            return False
        if not row.get("has_track2_label", False):
            return False
        rare_set = set(int(x) for x in self.options.rare_track2_indices)
        return int(row["track2_left"]) in rare_set or int(row["track2_right"]) in rare_set

    def _sample_aug_params(self, t: int, rare_factor: float = 1.0) -> dict:
        min_ratio = float(self.options.temporal_crop_min)
        if rare_factor > 1.0:
            min_ratio = max(0.55, min_ratio - 0.08 * (rare_factor - 1.0))
        if min_ratio < 1.0:
            crop = int(random.uniform(min_ratio, 1.0) * t)
            crop = max(8, min(crop, t))
            start = random.randint(0, t - crop)
        else:
            crop = t
            start = 0

        shift_max = max(0, int(round(float(self.options.temporal_shift_max) * rare_factor)))
        shift = int(random.randint(-shift_max, shift_max)) if shift_max > 0 else 0

        speed_min = float(self.options.speed_min)
        speed_max = float(self.options.speed_max)
        if speed_max < speed_min:
            speed_min, speed_max = speed_max, speed_min
        if rare_factor > 1.0:
            center = 0.5 * (speed_min + speed_max)
            half = 0.5 * (speed_max - speed_min) * rare_factor
            speed_min = max(0.5, center - half)
            speed_max = min(1.5, center + half)
        speed = float(random.uniform(speed_min, speed_max))

        scale_min = float(self.options.scale_min)
        scale_max = float(self.options.scale_max)
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        if rare_factor > 1.0:
            center = 0.5 * (scale_min + scale_max)
            half = 0.5 * (scale_max - scale_min) * rare_factor
            scale_min = max(0.8, center - half)
            scale_max = min(1.2, center + half)
        scale = float(random.uniform(scale_min, scale_max))

        rotate_deg = float(self.options.rotate_deg) * rare_factor
        rotate_rad = np.deg2rad(float(random.uniform(-rotate_deg, rotate_deg)))
        translate_std = float(self.options.translate_std) * rare_factor
        translate_x = float(np.random.normal(0.0, translate_std))
        translate_y = float(np.random.normal(0.0, translate_std))

        jitter_std = float(self.options.jitter_std) * rare_factor
        conf_drop_prob = min(0.6, float(self.options.conf_drop_prob) * rare_factor)

        return {
            "crop": int(crop),
            "start": int(start),
            "shift": shift,
            "speed": speed,
            "scale": scale,
            "rotate_rad": rotate_rad,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "jitter_std": jitter_std,
            "conf_drop_prob": conf_drop_prob,
        }

    def _apply_speed_aug(self, x: np.ndarray, speed: float) -> np.ndarray:
        if abs(float(speed) - 1.0) <= 1e-6:
            return x
        c, t, v, m = x.shape
        src_idx = np.arange(t, dtype=np.float32)
        tgt_idx = np.arange(t, dtype=np.float32)
        center = 0.5 * (t - 1)
        sample_idx = (tgt_idx - center) / float(speed) + center
        sample_idx = np.clip(sample_idx, 0.0, float(t - 1))
        out = np.zeros_like(x, dtype=np.float32)
        for ci in range(c):
            for vi in range(v):
                for mi in range(m):
                    out[ci, :, vi, mi] = np.interp(
                        sample_idx,
                        src_idx,
                        x[ci, :, vi, mi],
                    )
        return out.astype(np.float32)

    def _apply_temporal_aug(self, x: np.ndarray, params: dict) -> np.ndarray:
        """
        x: (C, T, V, M)
        """
        if not self.options.enable_time_aug:
            return x.copy()
        out = x.copy()
        _, t, _, _ = out.shape
        crop = int(params["crop"])
        start = int(params["start"])

        if crop < t:
            out = out[:, start : start + crop]
            out = _resample_input_channels(out, t)

        shift = int(params["shift"])
        out = np.roll(out, shift=shift, axis=1)
        out = self._apply_speed_aug(out, float(params["speed"]))
        return out.astype(np.float32)

    def _apply_spatial_aug(self, joint: np.ndarray, params: dict) -> np.ndarray:
        out = joint.copy()
        if not self.options.enable_spatial_aug:
            return out
        rad = float(params["rotate_rad"])
        cos_t = float(np.cos(rad))
        sin_t = float(np.sin(rad))
        x = out[0].copy()
        y = out[1].copy()
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y
        out[0] = x_new
        out[1] = y_new
        out[0] = out[0] * float(params["scale"]) + float(params["translate_x"])
        out[1] = out[1] * float(params["scale"]) + float(params["translate_y"])
        return out.astype(np.float32)

    def _augment_joint(self, joint: np.ndarray, params: dict) -> np.ndarray:
        out = self._apply_temporal_aug(joint, params)
        out = self._apply_spatial_aug(out, params)
        jitter_std = float(params.get("jitter_std", self.options.jitter_std))
        if jitter_std > 0:
            out[:2] += np.random.normal(0.0, jitter_std, size=out[:2].shape).astype(np.float32)
        return out.astype(np.float32)

    def _augment_confidence(self, confidence: np.ndarray, params: dict) -> np.ndarray:
        out = self._apply_temporal_aug(confidence, params)
        if self.options.enable_conf_aug:
            noise_std = float(self.options.conf_noise_std)
            if noise_std > 0:
                out += np.random.normal(0.0, noise_std, size=out.shape).astype(np.float32)

            drop_prob = float(params.get("conf_drop_prob", self.options.conf_drop_prob))
            if drop_prob > 0:
                drop = np.random.rand(*out.shape) < drop_prob
                out[drop] = 0.0

            low_boost = float(self.options.conf_low_boost)
            if low_boost > 0:
                low_prob = np.clip((1.0 - out) * low_boost, 0.0, 1.0)
                low_mask = np.random.rand(*out.shape) < low_prob
                out[low_mask] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _apply_lr_flip_joint(self, joint: np.ndarray) -> np.ndarray:
        out = joint.copy()
        out[0] = -out[0]
        out = out[:, :, self.lr_flip_index, :]
        return out.astype(np.float32)

    def _apply_lr_flip_confidence(self, confidence: np.ndarray) -> np.ndarray:
        out = confidence.copy()
        out = out[:, :, self.lr_flip_index, :]
        return out.astype(np.float32)

    def _swap_direction_lr(self, direction_idx: int) -> int:
        # 0: forward, 1: backward, 2: left, 3: right
        if direction_idx == 2:
            return 3
        if direction_idx == 3:
            return 2
        return direction_idx

    def _build_confidence(self, score: np.ndarray) -> np.ndarray:
        """
        score: (1, T, V, 1), raw keypoint_scores from pose model.
        return: (1, T, V, 1), normalized confidence in [0, 1]
        """
        if not self.options.use_score:
            return np.ones_like(score, dtype=np.float32)

        clip_min = float(self.options.score_clip_min)
        clip_max = float(self.options.score_clip_max)
        if clip_max < clip_min:
            clip_min, clip_max = clip_max, clip_min

        s = np.clip(score.astype(np.float32), clip_min, clip_max)
        denom = max(clip_max - clip_min, 1e-6)
        conf = (s - clip_min) / denom

        power = float(self.options.score_power)
        if abs(power - 1.0) > 1e-6:
            conf = np.power(np.clip(conf, 0.0, 1.0), power).astype(np.float32)

        if self.options.score_only_above_thr:
            mask = score >= float(self.options.score_thr)
            conf = np.where(mask, conf, 0.0)

        return np.clip(conf, 0.0, 1.0).astype(np.float32)

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
        score = data["score"].astype(np.float32) if "score" in data.files else None
        if score is None:
            score = np.ones((1, joint.shape[1], joint.shape[2], joint.shape[3]), dtype=np.float32)
        confidence = self._build_confidence(score)

        if self.options.use_pca:
            joint = self._apply_joint_pca(joint)
        expected_v = len(self.graph.connect_joint)
        if joint.shape[2] != expected_v:
            raise ValueError(
                "Cached keypoint dimension mismatch: "
                f"got V={joint.shape[2]}, expected V={expected_v}. "
                "Please rerun preprocess with --overwrite after changing keypoint_indices."
            )

        direction_idx = int(row["direction_idx"])

        t1_left = np.zeros((17,), dtype=np.float32)
        t1_right = np.zeros((17,), dtype=np.float32)
        if row["has_track1_label"]:
            t1_left[:] = np.asarray(row["track1_left"], dtype=np.float32)
            t1_right[:] = np.asarray(row["track1_right"], dtype=np.float32)

        t2_left = int(row["track2_left"]) if row["has_track2_label"] else 0
        t2_right = int(row["track2_right"]) if row["has_track2_label"] else 0

        rare_factor = float(self.options.rare_strength) if self._is_rare_track2_row(row) else 1.0
        do_flip = False
        if self.options.train and self.options.enable_lr_flip:
            flip_prob = float(self.options.flip_prob)
            if rare_factor > 1.0:
                flip_prob = min(1.0, flip_prob * rare_factor)
            do_flip = random.random() < flip_prob
        if do_flip:
            joint = self._apply_lr_flip_joint(joint)
            confidence = self._apply_lr_flip_confidence(confidence)
            direction_idx = self._swap_direction_lr(direction_idx)
            t1_left, t1_right = t1_right.copy(), t1_left.copy()
            t2_left, t2_right = int(t2_right), int(t2_left)

        if self.options.train:
            _, t, _, _ = joint.shape
            aug_main = self._sample_aug_params(t, rare_factor=rare_factor)
            aug_ssl = self._sample_aug_params(t, rare_factor=rare_factor)
            joint_main = self._augment_joint(joint, aug_main)
            joint_ssl = self._augment_joint(joint, aug_ssl)
            conf_main = self._augment_confidence(confidence, aug_main)
            conf_ssl = self._augment_confidence(confidence, aug_ssl)
        else:
            joint_main = joint
            joint_ssl = joint
            conf_main = confidence
            conf_ssl = confidence

        x = self._build_multi_input(joint_main)
        x_ssl = self._build_multi_input(joint_ssl)

        direction = np.zeros((4,), dtype=np.float32)
        direction[direction_idx] = 1.0

        t1_mask = 1.0 if row["has_track1_label"] else 0.0
        t2_mask = 1.0 if row["has_track2_label"] else 0.0

        sample = {
            "x": torch.from_numpy(x),
            "x_ssl": torch.from_numpy(x_ssl),
            "confidence": torch.from_numpy(conf_main),
            "confidence_ssl": torch.from_numpy(conf_ssl),
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
