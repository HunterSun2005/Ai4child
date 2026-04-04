import csv
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .constants import TRACK2_LABELS
from .data import AichildClipDataset, DatasetOptions, load_manifest
from .graph import AichildGraph

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - allows submission rendering without torch
    torch = None
    DataLoader = None


def _to_device(batch: dict, device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_fold_checkpoint_map(
    work_dir: str,
    folds: str,
    preferred_name: str,
    fallback_name: str = "best.pt",
) -> Dict[int, str]:
    if folds == "all":
        fold_ids = []
        for name in sorted(os.listdir(work_dir)):
            if not name.startswith("fold_"):
                continue
            suffix = name.split("fold_", 1)[1]
            if not suffix.isdigit():
                continue
            fold_ids.append(int(suffix))
        if not fold_ids:
            raise FileNotFoundError(f"No fold_* directories found under {work_dir}")
    else:
        fold_ids = [int(x.strip()) for x in folds.split(",") if x.strip()]
        if not fold_ids:
            raise ValueError("No valid fold ids parsed from --folds")

    checkpoint_map: Dict[int, str] = {}
    for fid in sorted(fold_ids):
        fold_dir = os.path.join(work_dir, f"fold_{fid}")
        candidates = [preferred_name]
        if fallback_name and fallback_name not in candidates:
            candidates.append(fallback_name)

        chosen = ""
        for name in candidates:
            p = os.path.join(fold_dir, name)
            if os.path.exists(p):
                chosen = p
                break

        if not chosen:
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fid}: tried {candidates} under {fold_dir}"
            )
        checkpoint_map[fid] = chosen

    return checkpoint_map


def _load_cv_fold_metrics(work_dir: str) -> Dict[int, dict]:
    summary_path = os.path.join(work_dir, "cv_summary.json")
    if not os.path.exists(summary_path):
        return {}
    try:
        payload = json.load(open(summary_path, "r", encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to read cv summary at %s: %s", summary_path, exc)
        return {}

    fold_metrics: Dict[int, dict] = {}
    for item in payload.get("folds", []):
        try:
            fid = int(item.get("fold", -1))
        except Exception:
            continue
        if fid > 0:
            fold_metrics[fid] = item
    return fold_metrics


def _select_topk_checkpoints(
    checkpoint_map: Dict[int, str],
    fold_metrics: Dict[int, dict],
    metric_key: str,
    topk: int,
    target_name: str,
) -> Dict[int, str]:
    if topk <= 0 or len(checkpoint_map) <= 1 or topk >= len(checkpoint_map):
        return checkpoint_map

    scored = []
    num_missing = 0
    for fid, path in checkpoint_map.items():
        score = None
        metric = fold_metrics.get(fid, {})
        value = metric.get(metric_key, None)
        if isinstance(value, (int, float)):
            score = float(value)
        if score is None:
            score = float("-inf")
            num_missing += 1
        scored.append((score, fid, path))

    if num_missing == len(checkpoint_map):
        logging.warning(
            "ensemble_topk=%d requested for %s but no fold metric '%s' found. Keep all folds.",
            topk,
            target_name,
            metric_key,
        )
        return checkpoint_map

    scored = sorted(scored, key=lambda x: (x[0], -x[1]), reverse=True)
    selected = scored[:topk]
    selected_ids = [fid for _, fid, _ in selected]
    logging.info(
        "Top-k fold selection | target=%s | metric=%s | topk=%d | selected=%s",
        target_name,
        metric_key,
        topk,
        selected_ids,
    )
    return {fid: path for _, fid, path in sorted(selected, key=lambda x: x[1])}


def _collect_test_rows(
    manifest: List[dict],
    task: str,
    track1_test_ids: set,
    track2_test_ids: set,
) -> List[dict]:
    rows = []
    for row in manifest:
        use_track1 = task in {"track1", "both"} and (
            row.get("is_track1_test", False) or row["subject_id"] in track1_test_ids
        )
        use_track2 = task in {"track2", "both"} and (
            row.get("is_track2_test", False) or row["subject_id"] in track2_test_ids
        )
        if use_track1 or use_track2:
            rows.append(row)
    return rows


def _mean_or_empty(items: List[np.ndarray], empty_dim: int) -> np.ndarray:
    if not items:
        return np.zeros((empty_dim,), dtype=np.float32)
    return np.mean(items, axis=0).astype(np.float32)


def _run_inference_for_checkpoints(
    config: dict,
    graph: AichildGraph,
    loader,
    checkpoint_map: Dict[int, str],
    device,
    collect_track1: bool,
    collect_track2: bool,
    use_score: bool,
) -> Tuple[List[Dict[int, dict]], List[str]]:
    from .model import MultiTaskEfficientGCN

    fold_results: List[Dict[int, dict]] = []
    used_checkpoints: List[str] = []

    for fid in sorted(checkpoint_map.keys()):
        ckpt_path = checkpoint_map[fid]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = MultiTaskEfficientGCN(config, graph).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        t1_left_by_subject = defaultdict(list)
        t1_right_by_subject = defaultdict(list)
        t2_left_by_subject = defaultdict(list)
        t2_right_by_subject = defaultdict(list)

        with torch.no_grad():
            desc = f"Predict fold_{fid}"
            for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
                batch = _to_device(batch, device)
                confidence = batch["confidence"].float() if use_score else None
                outputs = model(
                    batch["x"].float(),
                    batch["direction"].float(),
                    confidence,
                )

                p_t1_left = p_t1_right = None
                p_t2_left = p_t2_right = None
                if collect_track1:
                    p_t1_left = torch.sigmoid(outputs["track1_left"]).cpu().numpy()
                    p_t1_right = torch.sigmoid(outputs["track1_right"]).cpu().numpy()
                if collect_track2:
                    p_t2_left = torch.softmax(outputs["track2_left"], dim=1).cpu().numpy()
                    p_t2_right = torch.softmax(outputs["track2_right"], dim=1).cpu().numpy()

                bsz = batch["x"].shape[0]
                for i in range(bsz):
                    sid = int(batch["subject_id"][i].item())
                    if collect_track1:
                        t1_left_by_subject[sid].append(p_t1_left[i])
                        t1_right_by_subject[sid].append(p_t1_right[i])
                    if collect_track2:
                        t2_left_by_subject[sid].append(p_t2_left[i])
                        t2_right_by_subject[sid].append(p_t2_right[i])

        all_subjects = sorted(
            set(t1_left_by_subject.keys())
            | set(t1_right_by_subject.keys())
            | set(t2_left_by_subject.keys())
            | set(t2_right_by_subject.keys())
        )
        fold_subject_result: Dict[int, dict] = {}
        for sid in all_subjects:
            item = {}
            if collect_track1:
                item["track1_left_prob"] = _mean_or_empty(t1_left_by_subject[sid], empty_dim=17).tolist()
                item["track1_right_prob"] = _mean_or_empty(t1_right_by_subject[sid], empty_dim=17).tolist()
            if collect_track2:
                item["track2_left_prob"] = _mean_or_empty(t2_left_by_subject[sid], empty_dim=5).tolist()
                item["track2_right_prob"] = _mean_or_empty(t2_right_by_subject[sid], empty_dim=5).tolist()
            fold_subject_result[sid] = item

        fold_results.append(fold_subject_result)
        used_checkpoints.append(ckpt_path)

    return fold_results, used_checkpoints


def predict_multitask(
    config: dict,
    folds: str = "all",
    output_path: str = "",
    task: str = "both",
    checkpoint_policy: str = "separate",
    ensemble_topk: int = 0,
) -> Dict[str, dict]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for prediction.")

    if task not in {"track1", "track2", "both"}:
        raise ValueError(f"Unsupported task={task}")
    if checkpoint_policy not in {"shared", "separate"}:
        raise ValueError(f"Unsupported checkpoint_policy={checkpoint_policy}")
    if int(ensemble_topk) < 0:
        raise ValueError("ensemble_topk must be >= 0.")

    paths_cfg = config["paths"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    comp_cfg = config["competition"]
    pca_cfg = data_cfg.get("pca", {})
    score_cfg = data_cfg.get("score", {})
    use_pca = bool(pca_cfg.get("enabled", False))
    use_score = bool(score_cfg.get("enabled", False))
    pca_model_path = os.path.abspath(
        paths_cfg.get("pca_model_path", "") or os.path.join(paths_cfg["work_dir"], "pca_joint_model.npz")
    )

    track1_test_ids = set(int(x) for x in comp_cfg.get("track1_test_ids", []))
    track2_test_ids = set(int(x) for x in comp_cfg.get("track2_test_ids", []))
    track1_thr = float(comp_cfg.get("track1_threshold", 0.5))

    manifest = load_manifest(os.path.abspath(paths_cfg["manifest_path"]))
    test_rows = _collect_test_rows(manifest, task, track1_test_ids, track2_test_ids)
    if len(test_rows) == 0:
        raise ValueError(f"No test rows found in manifest for task={task}.")

    graph = AichildGraph(keypoint_indices=data_cfg["keypoint_indices"])
    ds = AichildClipDataset(
        test_rows,
        graph,
        DatasetOptions(
            inputs=data_cfg["inputs"],
            root_joint=int(data_cfg["root_joint"]),
            num_frame=int(data_cfg["num_frame"]),
            score_thr=float(data_cfg["score_thr"]),
            train=False,
            return_ssl=False,
            jitter_std=0.0,
            temporal_crop_min=1.0,
            use_pca=use_pca,
            pca_model_path=pca_model_path,
            use_score=use_score,
            score_clip_min=float(score_cfg.get("clip_min", 0.0)),
            score_clip_max=float(score_cfg.get("clip_max", 1.0)),
            score_power=float(score_cfg.get("power", 1.0)),
            score_only_above_thr=bool(score_cfg.get("only_above_thr", True)),
        ),
    )
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=max(1, int(train_cfg["num_workers"]) // 2),
        pin_memory=torch.cuda.is_available(),
    )

    work_dir = os.path.abspath(paths_cfg["work_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    track1_ckpt_map = {}
    track2_ckpt_map = {}
    if task in {"track1", "both"}:
        if checkpoint_policy == "shared":
            track1_ckpt_map = _resolve_fold_checkpoint_map(
                work_dir, folds, preferred_name="best_track2.pt", fallback_name="best.pt"
            )
        else:
            track1_ckpt_map = _resolve_fold_checkpoint_map(
                work_dir, folds, preferred_name="best_track1.pt", fallback_name="best.pt"
            )

    if task in {"track2", "both"}:
        track2_ckpt_map = _resolve_fold_checkpoint_map(
            work_dir, folds, preferred_name="best_track2.pt", fallback_name="best.pt"
        )

    if int(ensemble_topk) > 0:
        fold_metrics = _load_cv_fold_metrics(work_dir)
        if task in {"track1", "both"} and track1_ckpt_map:
            track1_metric = "best_track2_acc" if checkpoint_policy == "shared" else "best_track1_f1"
            track1_ckpt_map = _select_topk_checkpoints(
                track1_ckpt_map,
                fold_metrics,
                metric_key=track1_metric,
                topk=int(ensemble_topk),
                target_name="track1",
            )
        if task in {"track2", "both"} and track2_ckpt_map:
            track2_ckpt_map = _select_topk_checkpoints(
                track2_ckpt_map,
                fold_metrics,
                metric_key="best_track2_acc",
                topk=int(ensemble_topk),
                target_name="track2",
            )

    merged = defaultdict(
        lambda: {
            "track1_left_prob": [],
            "track1_right_prob": [],
            "track2_left_prob": [],
            "track2_right_prob": [],
        }
    )

    checkpoints_payload = {"track1": [], "track2": []}

    # If both tasks resolve to the exact same checkpoints, run one pass and collect both outputs.
    if task == "both" and track1_ckpt_map and track1_ckpt_map == track2_ckpt_map:
        fold_results, ckpts = _run_inference_for_checkpoints(
            config,
            graph,
            loader,
            track1_ckpt_map,
            device,
            collect_track1=True,
            collect_track2=True,
            use_score=use_score,
        )
        checkpoints_payload["track1"] = ckpts
        checkpoints_payload["track2"] = ckpts
        for fold_result in fold_results:
            for sid, item in fold_result.items():
                merged[sid]["track1_left_prob"].append(np.asarray(item["track1_left_prob"], dtype=np.float32))
                merged[sid]["track1_right_prob"].append(np.asarray(item["track1_right_prob"], dtype=np.float32))
                merged[sid]["track2_left_prob"].append(np.asarray(item["track2_left_prob"], dtype=np.float32))
                merged[sid]["track2_right_prob"].append(np.asarray(item["track2_right_prob"], dtype=np.float32))
    else:
        if task in {"track1", "both"}:
            fold_results_t1, ckpts_t1 = _run_inference_for_checkpoints(
                config,
                graph,
                loader,
                track1_ckpt_map,
                device,
                collect_track1=True,
                collect_track2=False,
                use_score=use_score,
            )
            checkpoints_payload["track1"] = ckpts_t1
            for fold_result in fold_results_t1:
                for sid, item in fold_result.items():
                    merged[sid]["track1_left_prob"].append(np.asarray(item["track1_left_prob"], dtype=np.float32))
                    merged[sid]["track1_right_prob"].append(np.asarray(item["track1_right_prob"], dtype=np.float32))

        if task in {"track2", "both"}:
            fold_results_t2, ckpts_t2 = _run_inference_for_checkpoints(
                config,
                graph,
                loader,
                track2_ckpt_map,
                device,
                collect_track1=False,
                collect_track2=True,
                use_score=use_score,
            )
            checkpoints_payload["track2"] = ckpts_t2
            for fold_result in fold_results_t2:
                for sid, item in fold_result.items():
                    merged[sid]["track2_left_prob"].append(np.asarray(item["track2_left_prob"], dtype=np.float32))
                    merged[sid]["track2_right_prob"].append(np.asarray(item["track2_right_prob"], dtype=np.float32))

    track1_predictions: Dict[str, dict] = {}
    track2_predictions: Dict[str, dict] = {}
    track1_subject_ids = {
        str(r["subject_id"])
        for r in manifest
        if r.get("is_track1_test", False) or r["subject_id"] in track1_test_ids
    }
    track2_subject_ids = {
        str(r["subject_id"])
        for r in manifest
        if r.get("is_track2_test", False) or r["subject_id"] in track2_test_ids
    }

    for sid in sorted(merged.keys()):
        sid_str = str(sid)

        t1_left_prob = _mean_or_empty(merged[sid]["track1_left_prob"], empty_dim=17)
        t1_right_prob = _mean_or_empty(merged[sid]["track1_right_prob"], empty_dim=17)
        t2_left_prob = _mean_or_empty(merged[sid]["track2_left_prob"], empty_dim=5)
        t2_right_prob = _mean_or_empty(merged[sid]["track2_right_prob"], empty_dim=5)

        if task in {"track1", "both"} and sid_str in track1_subject_ids:
            left_bin = (t1_left_prob >= track1_thr).astype(np.int64)
            right_bin = (t1_right_prob >= track1_thr).astype(np.int64)
            total = int(left_bin.sum() + right_bin.sum())

            track1_predictions[sid_str] = {
                "left_prob": t1_left_prob.tolist(),
                "right_prob": t1_right_prob.tolist(),
                "left_binary": left_bin.tolist(),
                "right_binary": right_bin.tolist(),
                "total": total,
                "threshold": track1_thr,
            }

        if task in {"track2", "both"} and sid_str in track2_subject_ids:
            left_idx = int(np.argmax(t2_left_prob))
            right_idx = int(np.argmax(t2_right_prob))
            track2_predictions[sid_str] = {
                "left_index": left_idx,
                "right_index": right_idx,
                "left_label": TRACK2_LABELS[left_idx],
                "right_label": TRACK2_LABELS[right_idx],
                "left_prob": t2_left_prob.tolist(),
                "right_prob": t2_right_prob.tolist(),
            }

    if not output_path:
        output_path = os.path.abspath(paths_cfg["prediction_path"])
    else:
        output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoints": checkpoints_payload,
                "task": task,
                "checkpoint_policy": checkpoint_policy,
                "ensemble_topk": int(ensemble_topk),
                "pca": {
                    "enabled": use_pca,
                    "model_path": pca_model_path if use_pca else "",
                },
                "score": {
                    "enabled": use_score,
                    "threshold": float(data_cfg["score_thr"]),
                    "clip_min": float(score_cfg.get("clip_min", 0.0)),
                    "clip_max": float(score_cfg.get("clip_max", 1.0)),
                    "power": float(score_cfg.get("power", 1.0)),
                    "only_above_thr": bool(score_cfg.get("only_above_thr", True)),
                },
                "predictions": {
                    "track1": track1_predictions,
                    "track2": track2_predictions,
                },
            },
            f,
            indent=2,
        )

    logging.info(
        "Predictions saved to %s | policy=%s | topk=%d | pca=%s | track1_subjects=%d | track2_subjects=%d",
        output_path,
        checkpoint_policy,
        int(ensemble_topk),
        use_pca,
        len(track1_predictions),
        len(track2_predictions),
    )
    return {
        "track1": track1_predictions,
        "track2": track2_predictions,
    }


def predict_track2(config: dict, folds: str = "all", output_path: str = "") -> Dict[str, dict]:
    """
    Backward-compatible helper:
    returns only track2 predictions while writing the new multitask json format.
    """
    predictions = predict_multitask(
        config,
        folds=folds,
        output_path=output_path,
        task="track2",
        checkpoint_policy="shared",
    )
    return predictions["track2"]


def _load_prediction_payload(prediction_json: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    with open(prediction_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # New format:
    # {
    #   "predictions": {
    #       "track1": {...},
    #       "track2": {...}
    #   }
    # }
    predictions = payload.get("predictions", {})
    if isinstance(predictions, dict) and "track1" in predictions and "track2" in predictions:
        return predictions.get("track1", {}), predictions.get("track2", {})

    # Old format (track2 only):
    # {
    #   "predictions": {
    #       "<sid>": {"left_label": ..., "right_label": ...}
    #   }
    # }
    if isinstance(predictions, dict):
        return {}, predictions

    raise ValueError("Invalid prediction json format.")


def make_submission_from_template(
    template_path: str,
    prediction_json: str,
    output_path: str,
) -> str:
    template_path = os.path.abspath(template_path)
    prediction_json = os.path.abspath(prediction_json)
    output_path = os.path.abspath(output_path)

    track1_preds, track2_preds = _load_prediction_payload(prediction_json)

    with open(template_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Submission template is empty.")

    # Some templates contain a trailing empty-header column.
    # Always drop unnamed columns to avoid Kaggle null-value errors.
    original_col_count = len(rows[0])
    keep_indices = []
    for col_idx, col_name in enumerate(rows[0]):
        col_name = col_name.strip()
        if col_name != "":
            keep_indices.append(col_idx)

    if len(keep_indices) != original_col_count:
        rows = [
            [row[i] if i < len(row) else "" for i in keep_indices]
            for row in rows
        ]
        logging.info(
            "Dropped %d unnamed columns from template.",
            original_col_count - len(keep_indices),
        )

    header = rows[0]
    id_idx = header.index("ID")
    left_subtype_idx = header.index("Left_gait_subtype")
    right_subtype_idx = header.index("Right_gait_subtype")

    left_indices = [header.index(f"L{i}") for i in range(1, 18)]
    right_indices = [header.index(f"R{i}") for i in range(1, 18)]
    total_idx = header.index("Total")
    missing_track1 = 0
    missing_track2 = 0

    for i in range(1, len(rows)):
        row = rows[i]
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))

        sample_id = row[id_idx].strip()

        if sample_id.startswith("track1-"):
            subject_id = sample_id.split("-", 1)[1]
            if subject_id not in track1_preds:
                missing_track1 += 1
                rows[i] = row
                continue

            pred = track1_preds[subject_id]
            left_bin = [int(x) for x in pred["left_binary"]]
            right_bin = [int(x) for x in pred["right_binary"]]
            if len(left_bin) != 17 or len(right_bin) != 17:
                raise ValueError(f"track1 prediction length must be 17+17 for {sample_id}")

            for j, col_idx in enumerate(left_indices):
                row[col_idx] = str(left_bin[j])
            for j, col_idx in enumerate(right_indices):
                row[col_idx] = str(right_bin[j])
            row[total_idx] = str(int(pred.get("total", sum(left_bin) + sum(right_bin))))

        elif sample_id.startswith("track2-"):
            subject_id = sample_id.split("-", 1)[1]
            if subject_id not in track2_preds:
                missing_track2 += 1
                rows[i] = row
                continue

            pred = track2_preds[subject_id]
            row[left_subtype_idx] = pred["left_label"]
            row[right_subtype_idx] = pred["right_label"]

        rows[i] = row

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    if missing_track1 > 0 or missing_track2 > 0:
        logging.warning(
            "Submission rendered with missing predictions: track1=%d, track2=%d",
            missing_track1,
            missing_track2,
        )

    logging.info("Submission written to %s", output_path)
    return output_path
