import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .constants import TRACK2_LABELS
from .data import AichildClipDataset, DatasetOptions, load_manifest
from .graph import AichildGraph
from .model import MultiTaskEfficientGCN


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_subject_folds(subject_ids: List[int], num_folds: int, seed: int) -> List[List[int]]:
    ids = list(sorted(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = [[] for _ in range(num_folds)]
    for i, sid in enumerate(ids):
        folds[i % num_folds].append(sid)
    return folds


def _track2_subject_labels(rows: List[dict]) -> Dict[int, Tuple[int, int]]:
    labels = {}
    for r in rows:
        if r.get("has_track2_label", False) and not r.get("is_track2_test", False):
            labels[int(r["subject_id"])] = (int(r["track2_left"]), int(r["track2_right"]))
    return labels


def _track2_subject_counts(
    subject_ids: List[int],
    subject_labels: Dict[int, Tuple[int, int]],
) -> Dict[str, Dict[int, int]]:
    left = Counter()
    right = Counter()
    for sid in subject_ids:
        if sid not in subject_labels:
            continue
        l, r = subject_labels[sid]
        left[l] += 1
        right[r] += 1
    return {
        "left": dict(sorted(left.items())),
        "right": dict(sorted(right.items())),
    }


def _split_track2_subject_folds(
    subject_labels: Dict[int, Tuple[int, int]],
    num_folds: int,
    seed: int,
    singleton_min_count: int,
) -> Tuple[List[List[int]], List[int]]:
    if num_folds < 2:
        raise ValueError("Track2 CV requires at least 2 folds.")

    all_subjects = sorted(subject_labels.keys())
    left_counts = Counter(l for l, _ in subject_labels.values())
    right_counts = Counter(r for _, r in subject_labels.values())

    fixed_train_subjects = []
    fold_candidates = []
    for sid in all_subjects:
        l, r = subject_labels[sid]
        if left_counts[l] < singleton_min_count or right_counts[r] < singleton_min_count:
            fixed_train_subjects.append(sid)
        else:
            fold_candidates.append(sid)

    rng = random.Random(seed)
    rng.shuffle(fold_candidates)
    fold_candidates.sort(
        key=lambda sid: (
            min(left_counts[subject_labels[sid][0]], right_counts[subject_labels[sid][1]]),
            rng.random(),
        )
    )

    folds = [[] for _ in range(num_folds)]
    fold_left = [Counter() for _ in range(num_folds)]
    fold_right = [Counter() for _ in range(num_folds)]
    for sid in fold_candidates:
        l, r = subject_labels[sid]
        best_idx = min(
            range(num_folds),
            key=lambda i: (
                len(folds[i]),
                fold_left[i][l] + fold_right[i][r],
                rng.random(),
            ),
        )
        folds[best_idx].append(sid)
        fold_left[best_idx][l] += 1
        fold_right[best_idx][r] += 1

    return [sorted(f) for f in folds], sorted(fixed_train_subjects)


def _track2_fold_diagnostics(
    train_subjects: List[int],
    val_subjects: List[int],
    subject_labels: Dict[int, Tuple[int, int]],
    fixed_train_subjects: List[int],
) -> Dict[str, object]:
    train_counts = _track2_subject_counts(train_subjects, subject_labels)
    val_counts = _track2_subject_counts(val_subjects, subject_labels)
    unseen_left = sorted(set(val_counts["left"]) - set(train_counts["left"]))
    unseen_right = sorted(set(val_counts["right"]) - set(train_counts["right"]))
    return {
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "fixed_train_subjects": sorted(fixed_train_subjects),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "unseen_left": unseen_left,
        "unseen_right": unseen_right,
    }


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def _track2_class_weight_tensor(train_cfg: dict, device: torch.device) -> Optional[torch.Tensor]:
    if not bool(train_cfg.get("use_track2_loss_weights", False)):
        return None
    class_weights_cfg = train_cfg.get("track2_class_weights", {})
    default_w = float(train_cfg.get("track2_sampler_default_weight", 1.0))
    weights = [
        float(class_weights_cfg.get(str(i), default_w))
        for i in range(len(TRACK2_LABELS))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _track2_sampler_weights(train_rows: List[dict], train_cfg: dict) -> torch.Tensor:
    class_weights_cfg = train_cfg.get("track2_class_weights", {})
    default_w = float(train_cfg.get("track2_sampler_default_weight", 1.0))
    clips_per_subject = Counter(int(r["subject_id"]) for r in train_rows if r.get("has_track2_label", False))
    weights = []
    for r in train_rows:
        l = int(r["track2_left"])
        rr = int(r["track2_right"])
        lw = float(class_weights_cfg.get(str(l), default_w))
        rw = float(class_weights_cfg.get(str(rr), default_w))
        subject_clip_count = max(1, clips_per_subject[int(r["subject_id"])])
        weights.append(max(1e-6, 0.5 * (lw + rw) / subject_clip_count))
    return torch.as_tensor(weights, dtype=torch.double)


def _make_loader(
    rows: List[dict],
    graph: AichildGraph,
    opts: DatasetOptions,
    train_cfg: dict,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
) -> Optional[DataLoader]:
    if not rows:
        return None
    if num_workers is None:
        num_workers = int(train_cfg["num_workers"])
    dataset = AichildClipDataset(rows, graph, opts)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _next_batch(loader: DataLoader, iter_state: dict, key: str) -> dict:
    iterator = iter_state.get(key)
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    iter_state[key] = iterator
    return batch


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.size(0)
    if batch_size < 2:
        return z1.sum() * 0.0
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _evaluate_track2_subject_level(model, loader, device, use_score: bool) -> Dict[str, float]:
    model.eval()
    subject_probs_left = defaultdict(list)
    subject_probs_right = defaultdict(list)
    subject_gt_left = {}
    subject_gt_right = {}

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            confidence = batch["confidence"].float() if use_score else None
            outputs = model(
                batch["x"].float(),
                batch["direction"].float(),
                confidence,
            )
            probs_left = torch.softmax(outputs["track2_left"], dim=1)
            probs_right = torch.softmax(outputs["track2_right"], dim=1)

            for i in range(probs_left.size(0)):
                if batch["track2_mask"][i].item() <= 0:
                    continue
                sid = int(batch["subject_id"][i].item())
                subject_probs_left[sid].append(probs_left[i].detach().cpu().numpy())
                subject_probs_right[sid].append(probs_right[i].detach().cpu().numpy())
                subject_gt_left[sid] = int(batch["track2_left"][i].item())
                subject_gt_right[sid] = int(batch["track2_right"][i].item())

    if not subject_probs_left:
        return {
            "num_subject": 0,
            "left_acc": 0.0,
            "right_acc": 0.0,
            "mean_acc": 0.0,
        }

    left_hits = 0
    right_hits = 0
    for sid in sorted(subject_probs_left.keys()):
        pred_left = int(np.argmax(np.mean(subject_probs_left[sid], axis=0)))
        pred_right = int(np.argmax(np.mean(subject_probs_right[sid], axis=0)))
        left_hits += int(pred_left == subject_gt_left[sid])
        right_hits += int(pred_right == subject_gt_right[sid])

    n = len(subject_probs_left)
    left_acc = left_hits / n
    right_acc = right_hits / n
    return {
        "num_subject": n,
        "left_acc": left_acc,
        "right_acc": right_acc,
        "mean_acc": 0.5 * (left_acc + right_acc),
    }


def _binary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return float((2 * tp) / denom)


def _evaluate_track1_subject_level(
    model,
    loader,
    device,
    threshold: float,
    use_score: bool,
) -> Dict[str, float]:
    model.eval()
    subject_probs_left = defaultdict(list)
    subject_probs_right = defaultdict(list)
    subject_gt_left = {}
    subject_gt_right = {}

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            confidence = batch["confidence"].float() if use_score else None
            outputs = model(
                batch["x"].float(),
                batch["direction"].float(),
                confidence,
            )
            probs_left = torch.sigmoid(outputs["track1_left"])
            probs_right = torch.sigmoid(outputs["track1_right"])

            for i in range(probs_left.size(0)):
                if batch["track1_mask"][i].item() <= 0:
                    continue
                sid = int(batch["subject_id"][i].item())
                subject_probs_left[sid].append(probs_left[i].detach().cpu().numpy())
                subject_probs_right[sid].append(probs_right[i].detach().cpu().numpy())
                subject_gt_left[sid] = batch["track1_left"][i].detach().cpu().numpy().astype(np.int64)
                subject_gt_right[sid] = batch["track1_right"][i].detach().cpu().numpy().astype(np.int64)

    if not subject_probs_left:
        return {
            "num_subject": 0,
            "left_f1": 0.0,
            "right_f1": 0.0,
            "mean_f1": 0.0,
            "left_acc": 0.0,
            "right_acc": 0.0,
            "mean_acc": 0.0,
        }

    left_f1_list = []
    right_f1_list = []
    left_acc_list = []
    right_acc_list = []
    for sid in sorted(subject_probs_left.keys()):
        pred_left = (np.mean(subject_probs_left[sid], axis=0) >= threshold).astype(np.int64)
        pred_right = (np.mean(subject_probs_right[sid], axis=0) >= threshold).astype(np.int64)
        gt_left = subject_gt_left[sid]
        gt_right = subject_gt_right[sid]

        left_f1_list.append(_binary_f1(pred_left, gt_left))
        right_f1_list.append(_binary_f1(pred_right, gt_right))
        left_acc_list.append(float((pred_left == gt_left).mean()))
        right_acc_list.append(float((pred_right == gt_right).mean()))

    left_f1 = float(np.mean(left_f1_list))
    right_f1 = float(np.mean(right_f1_list))
    left_acc = float(np.mean(left_acc_list))
    right_acc = float(np.mean(right_acc_list))
    return {
        "num_subject": len(left_f1_list),
        "left_f1": left_f1,
        "right_f1": right_f1,
        "mean_f1": 0.5 * (left_f1 + right_f1),
        "left_acc": left_acc,
        "right_acc": right_acc,
        "mean_acc": 0.5 * (left_acc + right_acc),
    }


def train_cv(config: dict, cv_folds: int, max_epochs: int = -1) -> Dict[str, object]:
    _seed_all(int(config["train"]["seed"]))

    paths_cfg = config["paths"]
    train_cfg = config["train"]
    aug_cfg = train_cfg.get("augment", {})
    data_cfg = config["data"]
    pca_cfg = data_cfg.get("pca", {})
    score_cfg = data_cfg.get("score", {})
    comp_cfg = config.get("competition", {})
    track1_threshold = float(comp_cfg.get("track1_threshold", 0.5))
    task_loader_mode = str(train_cfg.get("task_loader_mode", "separate"))
    if task_loader_mode != "separate":
        raise ValueError(f"Unsupported task_loader_mode: {task_loader_mode}")
    use_pca = bool(pca_cfg.get("enabled", False))
    use_score = bool(score_cfg.get("enabled", False))
    pca_model_path = os.path.abspath(
        paths_cfg.get("pca_model_path", "") or os.path.join(paths_cfg["work_dir"], "pca_joint_model.npz")
    )

    manifest_path = os.path.abspath(paths_cfg["manifest_path"])
    work_dir = os.path.abspath(paths_cfg["work_dir"])
    os.makedirs(work_dir, exist_ok=True)

    rows = load_manifest(manifest_path)

    track2_subject_labels = _track2_subject_labels(rows)
    track2_train_subjects = sorted(track2_subject_labels.keys())
    if len(track2_train_subjects) == 0:
        raise ValueError("No track2 train subjects found in manifest.")

    cv_strategy = str(train_cfg.get("track2_cv_strategy", "stratified_holdout_singletons"))
    if cv_strategy == "stratified_holdout_singletons":
        folds, fixed_train_subjects = _split_track2_subject_folds(
            track2_subject_labels,
            cv_folds,
            int(train_cfg["seed"]),
            int(train_cfg.get("track2_singleton_min_count", 2)),
        )
    elif cv_strategy == "random":
        folds = _split_subject_folds(track2_train_subjects, cv_folds, int(train_cfg["seed"]))
        fixed_train_subjects = []
    else:
        raise ValueError(f"Unknown track2_cv_strategy: {cv_strategy}")

    graph = AichildGraph(keypoint_indices=data_cfg["keypoint_indices"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training device: %s", device)
    logging.info("PCA setting | enabled=%s | model=%s", use_pca, pca_model_path if use_pca else "<disabled>")
    logging.info(
        "Score setting | enabled=%s | thr=%.4f | clip=[%.3f, %.3f] | power=%.3f | only_above_thr=%s",
        use_score,
        float(data_cfg["score_thr"]),
        float(score_cfg.get("clip_min", 0.0)),
        float(score_cfg.get("clip_max", 1.0)),
        float(score_cfg.get("power", 1.0)),
        bool(score_cfg.get("only_above_thr", True)),
    )
    logging.info(
        "Aug setting | time=%s | spatial=%s | conf=%s | flip=%s(%.2f) | rare=%s(str=%.2f)",
        bool(aug_cfg.get("enable_time_aug", True)),
        bool(aug_cfg.get("enable_spatial_aug", True)),
        bool(aug_cfg.get("enable_conf_aug", True)),
        bool(aug_cfg.get("enable_lr_flip", False)),
        float(aug_cfg.get("flip_prob", 0.0)),
        bool(aug_cfg.get("enable_rare_aug", True)),
        float(aug_cfg.get("rare_strength", 1.5)),
    )
    logging.info(
        "Task loader setting | mode=%s | track2_cv=%s | fixed_train_subjects=%s",
        task_loader_mode,
        cv_strategy,
        fixed_train_subjects,
    )

    fold_summaries = []
    for fold_idx, val_subjects in enumerate(folds, start=1):
        val_subjects = set(val_subjects)
        train_subjects = sorted(set(track2_train_subjects) - val_subjects)
        fold_diag = _track2_fold_diagnostics(
            train_subjects,
            sorted(val_subjects),
            track2_subject_labels,
            fixed_train_subjects,
        )
        if fold_diag["unseen_left"] or fold_diag["unseen_right"]:
            logging.warning(
                "Fold %d has unseen Track2 classes | left=%s | right=%s",
                fold_idx,
                fold_diag["unseen_left"],
                fold_diag["unseen_right"],
            )

        train_rows = [r for r in rows if r["subject_id"] not in val_subjects]
        track2_train_rows = [
            r
            for r in train_rows
            if r.get("has_track2_label", False) and not r.get("is_track2_test", False)
        ]
        track1_train_rows = [r for r in train_rows if r.get("has_track1_label", False)]
        ssl_train_rows = train_rows
        val_rows = [r for r in rows if r["subject_id"] in val_subjects and r["has_track2_label"]]

        train_opts = DatasetOptions(
            inputs=data_cfg["inputs"],
            root_joint=int(data_cfg["root_joint"]),
            num_frame=int(data_cfg["num_frame"]),
            score_thr=float(data_cfg["score_thr"]),
            train=True,
            return_ssl=True,
            jitter_std=float(train_cfg["jitter_std"]),
            temporal_crop_min=float(train_cfg["temporal_crop_min"]),
            use_pca=use_pca,
            pca_model_path=pca_model_path,
            use_score=use_score,
            score_clip_min=float(score_cfg.get("clip_min", 0.0)),
            score_clip_max=float(score_cfg.get("clip_max", 1.0)),
            score_power=float(score_cfg.get("power", 1.0)),
            score_only_above_thr=bool(score_cfg.get("only_above_thr", True)),
            enable_time_aug=bool(aug_cfg.get("enable_time_aug", True)),
            temporal_shift_max=int(aug_cfg.get("temporal_shift_max", 8)),
            speed_min=float(aug_cfg.get("speed_min", 0.9)),
            speed_max=float(aug_cfg.get("speed_max", 1.1)),
            enable_spatial_aug=bool(aug_cfg.get("enable_spatial_aug", True)),
            rotate_deg=float(aug_cfg.get("rotate_deg", 6.0)),
            translate_std=float(aug_cfg.get("translate_std", 0.01)),
            scale_min=float(aug_cfg.get("scale_min", 0.95)),
            scale_max=float(aug_cfg.get("scale_max", 1.05)),
            enable_conf_aug=bool(aug_cfg.get("enable_conf_aug", True)),
            conf_noise_std=float(aug_cfg.get("conf_noise_std", 0.02)),
            conf_drop_prob=float(aug_cfg.get("conf_drop_prob", 0.05)),
            conf_low_boost=float(aug_cfg.get("conf_low_boost", 0.20)),
            enable_lr_flip=bool(aug_cfg.get("enable_lr_flip", False)),
            flip_prob=float(aug_cfg.get("flip_prob", 0.0)),
            enable_rare_aug=bool(aug_cfg.get("enable_rare_aug", True)),
            rare_track2_indices=tuple(int(x) for x in aug_cfg.get("rare_track2_indices", [3, 4])),
            rare_strength=float(aug_cfg.get("rare_strength", 1.5)),
        )
        val_opts = DatasetOptions(
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
            enable_time_aug=False,
            enable_spatial_aug=False,
            enable_conf_aug=False,
            enable_lr_flip=False,
            enable_rare_aug=False,
        )

        val_ds = AichildClipDataset(val_rows, graph, val_opts)
        track2_train_eval_ds = AichildClipDataset(track2_train_rows, graph, val_opts)

        track2_sampler = None
        if track2_train_rows and bool(train_cfg.get("use_track2_weighted_sampler", True)):
            track2_weights = _track2_sampler_weights(track2_train_rows, train_cfg)
            track2_sampler = WeightedRandomSampler(
                weights=track2_weights,
                num_samples=len(track2_weights),
                replacement=True,
            )

        track2_loader = _make_loader(
            track2_train_rows,
            graph,
            train_opts,
            train_cfg,
            sampler=track2_sampler,
            shuffle=(track2_sampler is None),
        )
        track1_loader = _make_loader(
            track1_train_rows,
            graph,
            train_opts,
            train_cfg,
            shuffle=True,
        )
        ssl_loader = _make_loader(
            ssl_train_rows,
            graph,
            train_opts,
            train_cfg,
            shuffle=True,
        )
        if track2_loader is None and float(train_cfg["lambda_t2"]) > 0:
            raise ValueError(f"Fold {fold_idx} has no Track2 training clips.")

        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=max(1, int(train_cfg["num_workers"]) // 2),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        track2_train_eval_loader = DataLoader(
            track2_train_eval_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=max(1, int(train_cfg["num_workers"]) // 2),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = MultiTaskEfficientGCN(config, graph).to(device)
        track2_ce_weight = _track2_class_weight_tensor(train_cfg, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

        epochs = int(train_cfg["epochs"] if max_epochs <= 0 else max_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        fold_dir = os.path.join(work_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        best_track2_acc = -1.0
        best_track1_f1 = -1.0
        best_state_track2 = None
        best_state_track1 = None
        history = []
        lambda_ssl = float(train_cfg["lambda_ssl"])
        lambda_t1 = float(train_cfg["lambda_t1"])
        lambda_t2 = float(train_cfg["lambda_t2"])

        logging.info(
            "Fold %d/%d | train clips=%d | t1 clips=%d | t2 clips=%d | ssl clips=%d | "
            "val clips=%d | val subjects=%d | counts=%s",
            fold_idx,
            cv_folds,
            len(train_rows),
            len(track1_train_rows),
            len(track2_train_rows),
            len(ssl_train_rows),
            len(val_rows),
            len(val_subjects),
            fold_diag,
        )

        for epoch in range(1, epochs + 1):
            model.train()
            enabled_loader_lengths = []
            if lambda_t2 > 0 and track2_loader is not None:
                enabled_loader_lengths.append(len(track2_loader))
            if lambda_t1 > 0 and track1_loader is not None:
                enabled_loader_lengths.append(len(track1_loader))
            if lambda_ssl > 0 and ssl_loader is not None:
                enabled_loader_lengths.append(len(ssl_loader))
            if not enabled_loader_lengths:
                raise ValueError("No enabled task loaders. Check lambda_ssl/lambda_t1/lambda_t2 and manifest labels.")
            epoch_steps = max(enabled_loader_lengths)

            running = {
                "loss": 0.0,
                "ssl": 0.0,
                "t1": 0.0,
                "t2": 0.0,
                "t2_mask_mean": 0.0,
                "t2_labeled_samples": 0.0,
                "steps": 0,
            }
            t2_left_seen = Counter()
            t2_right_seen = Counter()
            iter_state = {}

            pbar = tqdm(range(epoch_steps), desc=f"Fold{fold_idx} Epoch{epoch}", dynamic_ncols=True)
            for _ in pbar:
                loss_terms = []
                loss_ssl = torch.zeros((), device=device)
                loss_t1 = torch.zeros((), device=device)
                loss_t2 = torch.zeros((), device=device)

                if lambda_t2 > 0 and track2_loader is not None:
                    batch_t2 = _to_device(_next_batch(track2_loader, iter_state, "track2"), device)
                    conf_t2 = batch_t2["confidence"].float() if use_score else None
                    outputs_t2 = model(
                        batch_t2["x"].float(),
                        batch_t2["direction"].float(),
                        conf_t2,
                    )
                    t2_mask = batch_t2["track2_mask"].float()
                    loss_t2 = _masked_ce(
                        outputs_t2["track2_left"],
                        batch_t2["track2_left"].long(),
                        t2_mask,
                        weight=track2_ce_weight,
                    )
                    loss_t2 += _masked_ce(
                        outputs_t2["track2_right"],
                        batch_t2["track2_right"].long(),
                        t2_mask,
                        weight=track2_ce_weight,
                    )
                    loss_terms.append(lambda_t2 * loss_t2)

                    valid_t2 = t2_mask > 0
                    running["t2_mask_mean"] += float(t2_mask.mean().item())
                    running["t2_labeled_samples"] += float(t2_mask.sum().item())
                    t2_left_seen.update(int(x) for x in batch_t2["track2_left"][valid_t2].detach().cpu().tolist())
                    t2_right_seen.update(int(x) for x in batch_t2["track2_right"][valid_t2].detach().cpu().tolist())

                if lambda_t1 > 0 and track1_loader is not None:
                    batch_t1 = _to_device(_next_batch(track1_loader, iter_state, "track1"), device)
                    conf_t1 = batch_t1["confidence"].float() if use_score else None
                    outputs_t1 = model(
                        batch_t1["x"].float(),
                        batch_t1["direction"].float(),
                        conf_t1,
                    )
                    t1_mask = batch_t1["track1_mask"].float()
                    loss_t1 = _masked_bce(outputs_t1["track1_left"], batch_t1["track1_left"].float(), t1_mask)
                    loss_t1 += _masked_bce(outputs_t1["track1_right"], batch_t1["track1_right"].float(), t1_mask)
                    loss_terms.append(lambda_t1 * loss_t1)

                if lambda_ssl > 0 and ssl_loader is not None:
                    batch_ssl = _to_device(_next_batch(ssl_loader, iter_state, "ssl"), device)
                    x = batch_ssl["x"].float()
                    x_ssl = batch_ssl["x_ssl"].float()
                    conf = batch_ssl["confidence"].float() if use_score else None
                    conf_ssl = batch_ssl["confidence_ssl"].float() if use_score else None
                    direction = batch_ssl["direction"].float()
                    z1 = model.ssl_embedding(x, direction, conf)
                    z2 = model.ssl_embedding(x_ssl, direction, conf_ssl)
                    loss_ssl = _info_nce(z1, z2, float(train_cfg["ssl_temperature"]))
                    loss_terms.append(lambda_ssl * loss_ssl)

                loss = sum(loss_terms)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))
                optimizer.step()

                running["loss"] += float(loss.item())
                running["ssl"] += float(loss_ssl.item())
                running["t1"] += float(loss_t1.item())
                running["t2"] += float(loss_t2.item())
                running["steps"] += 1

                pbar.set_postfix(
                    loss=f"{running['loss'] / max(running['steps'], 1):.4f}",
                    t2=f"{running['t2'] / max(running['steps'], 1):.4f}",
                )

            scheduler.step()

            train_metrics_t2 = _evaluate_track2_subject_level(
                model,
                track2_train_eval_loader,
                device,
                use_score=use_score,
            )
            val_metrics_t2 = _evaluate_track2_subject_level(
                model,
                val_loader,
                device,
                use_score=use_score,
            )
            val_metrics_t1 = _evaluate_track1_subject_level(
                model,
                val_loader,
                device,
                threshold=track1_threshold,
                use_score=use_score,
            )
            mean_loss = running["loss"] / max(running["steps"], 1)
            lr = float(optimizer.param_groups[0]["lr"])

            record = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_ssl": running["ssl"] / max(running["steps"], 1),
                "train_t1": running["t1"] / max(running["steps"], 1),
                "train_t2": running["t2"] / max(running["steps"], 1),
                "train_t2_mask_mean": running["t2_mask_mean"] / max(running["steps"], 1),
                "train_t2_labeled_samples": running["t2_labeled_samples"],
                "train_t2_seen_left": dict(sorted(t2_left_seen.items())),
                "train_t2_seen_right": dict(sorted(t2_right_seen.items())),
                "train_t2_mean_acc": train_metrics_t2["mean_acc"],
                "train_t2_left_acc": train_metrics_t2["left_acc"],
                "train_t2_right_acc": train_metrics_t2["right_acc"],
                "val_t2_mean_acc": val_metrics_t2["mean_acc"],
                "val_t2_left_acc": val_metrics_t2["left_acc"],
                "val_t2_right_acc": val_metrics_t2["right_acc"],
                "val_t1_mean_f1": val_metrics_t1["mean_f1"],
                "val_t1_left_f1": val_metrics_t1["left_f1"],
                "val_t1_right_f1": val_metrics_t1["right_f1"],
                "val_t1_mean_acc": val_metrics_t1["mean_acc"],
                "lr": lr,
            }
            history.append(record)

            logging.info(
                "Fold %d Epoch %d | loss=%.4f | "
                "train_t2_acc=%.4f | val_t2_acc=%.4f (L=%.4f, R=%.4f) | "
                "t2_seen=%s/%s | "
                "val_t1_f1=%.4f (L=%.4f, R=%.4f)",
                fold_idx,
                epoch,
                record["train_loss"],
                record["train_t2_mean_acc"],
                record["val_t2_mean_acc"],
                record["val_t2_left_acc"],
                record["val_t2_right_acc"],
                record["train_t2_seen_left"],
                record["train_t2_seen_right"],
                record["val_t1_mean_f1"],
                record["val_t1_left_f1"],
                record["val_t1_right_f1"],
            )

            if val_metrics_t2["mean_acc"] > best_track2_acc:
                best_track2_acc = val_metrics_t2["mean_acc"]
                best_state_track2 = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics_t2": val_metrics_t2,
                    "val_metrics_t1": val_metrics_t1,
                    "config": config,
                }
                torch.save(best_state_track2, os.path.join(fold_dir, "best_track2.pt"))
                # Keep backward compatibility with previous inference defaults.
                torch.save(best_state_track2, os.path.join(fold_dir, "best.pt"))

            if val_metrics_t1["mean_f1"] > best_track1_f1:
                best_track1_f1 = val_metrics_t1["mean_f1"]
                best_state_track1 = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics_t2": val_metrics_t2,
                    "val_metrics_t1": val_metrics_t1,
                    "config": config,
                }
                torch.save(best_state_track1, os.path.join(fold_dir, "best_track1.pt"))

        with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        fold_summary = {
            "fold": fold_idx,
            "best_track2_acc": best_track2_acc,
            "best_track2_epoch": int(best_state_track2["epoch"]) if best_state_track2 else -1,
            "best_track1_f1": best_track1_f1,
            "best_track1_epoch": int(best_state_track1["epoch"]) if best_state_track1 else -1,
            "checkpoint_track2": os.path.join(fold_dir, "best_track2.pt"),
            "checkpoint_track1": os.path.join(fold_dir, "best_track1.pt"),
            "checkpoint_compat": os.path.join(fold_dir, "best.pt"),
            "diagnostics": fold_diag,
        }
        fold_summaries.append(fold_summary)

    cv_result = {
        "num_folds": cv_folds,
        "folds": fold_summaries,
        "mean_best_track2_acc": (
            float(np.mean([f["best_track2_acc"] for f in fold_summaries])) if fold_summaries else 0.0
        ),
        "mean_best_track1_f1": (
            float(np.mean([f["best_track1_f1"] for f in fold_summaries])) if fold_summaries else 0.0
        ),
        # Backward-compatible key.
        "mean_best_acc": (
            float(np.mean([f["best_track2_acc"] for f in fold_summaries])) if fold_summaries else 0.0
        ),
    }

    with open(os.path.join(work_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_result, f, indent=2)

    logging.info("CV done: %s", cv_result)
    return cv_result
