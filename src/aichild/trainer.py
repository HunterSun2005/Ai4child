import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


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


def _evaluate_subject_level(model, loader, device) -> Dict[str, float]:
    model.eval()
    subject_probs_left = defaultdict(list)
    subject_probs_right = defaultdict(list)
    subject_gt_left = {}
    subject_gt_right = {}

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["x"].float(), batch["direction"].float())
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


def train_cv(config: dict, cv_folds: int, max_epochs: int = -1) -> Dict[str, object]:
    _seed_all(int(config["train"]["seed"]))

    paths_cfg = config["paths"]
    train_cfg = config["train"]
    data_cfg = config["data"]

    manifest_path = os.path.abspath(paths_cfg["manifest_path"])
    work_dir = os.path.abspath(paths_cfg["work_dir"])
    os.makedirs(work_dir, exist_ok=True)

    rows = load_manifest(manifest_path)

    track2_train_subjects = sorted(
        {
            r["subject_id"]
            for r in rows
            if r["has_track2_label"] and not r["is_track2_test"]
        }
    )
    if len(track2_train_subjects) == 0:
        raise ValueError("No track2 train subjects found in manifest.")

    folds = _split_subject_folds(track2_train_subjects, cv_folds, int(train_cfg["seed"]))
    graph = AichildGraph()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training device: %s", device)

    fold_summaries = []
    for fold_idx, val_subjects in enumerate(folds, start=1):
        val_subjects = set(val_subjects)
        train_rows = [r for r in rows if r["subject_id"] not in val_subjects]
        val_rows = [r for r in rows if r["subject_id"] in val_subjects and r["has_track2_label"]]

        train_opts = DatasetOptions(
            inputs=data_cfg["inputs"],
            root_joint=int(data_cfg["root_joint"]),
            num_frame=int(data_cfg["num_frame"]),
            train=True,
            return_ssl=True,
            jitter_std=float(train_cfg["jitter_std"]),
            temporal_crop_min=float(train_cfg["temporal_crop_min"]),
        )
        val_opts = DatasetOptions(
            inputs=data_cfg["inputs"],
            root_joint=int(data_cfg["root_joint"]),
            num_frame=int(data_cfg["num_frame"]),
            train=False,
            return_ssl=False,
            jitter_std=0.0,
            temporal_crop_min=1.0,
        )

        train_ds = AichildClipDataset(train_rows, graph, train_opts)
        val_ds = AichildClipDataset(val_rows, graph, val_opts)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(train_cfg["num_workers"]),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=max(1, int(train_cfg["num_workers"]) // 2),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = MultiTaskEfficientGCN(config, graph).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

        epochs = int(train_cfg["epochs"] if max_epochs <= 0 else max_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        fold_dir = os.path.join(work_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        best_acc = -1.0
        best_state = None
        history = []

        logging.info(
            "Fold %d/%d | train clips=%d | val clips=%d | val subjects=%d",
            fold_idx,
            cv_folds,
            len(train_rows),
            len(val_rows),
            len(val_subjects),
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running = {
                "loss": 0.0,
                "ssl": 0.0,
                "t1": 0.0,
                "t2": 0.0,
                "steps": 0,
            }

            pbar = tqdm(train_loader, desc=f"Fold{fold_idx} Epoch{epoch}", dynamic_ncols=True)
            for batch in pbar:
                batch = _to_device(batch, device)

                x = batch["x"].float()
                x_ssl = batch["x_ssl"].float()
                direction = batch["direction"].float()

                outputs = model(x, direction)

                t1_mask = batch["track1_mask"].float()
                t2_mask = batch["track2_mask"].float()

                loss_t1 = _masked_bce(outputs["track1_left"], batch["track1_left"].float(), t1_mask)
                loss_t1 += _masked_bce(outputs["track1_right"], batch["track1_right"].float(), t1_mask)

                loss_t2 = _masked_ce(outputs["track2_left"], batch["track2_left"].long(), t2_mask)
                loss_t2 += _masked_ce(outputs["track2_right"], batch["track2_right"].long(), t2_mask)

                z1 = model.ssl_embedding(x, direction)
                z2 = model.ssl_embedding(x_ssl, direction)
                loss_ssl = _info_nce(z1, z2, float(train_cfg["ssl_temperature"]))

                loss = (
                    float(train_cfg["lambda_ssl"]) * loss_ssl
                    + float(train_cfg["lambda_t1"]) * loss_t1
                    + float(train_cfg["lambda_t2"]) * loss_t2
                )

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

            val_metrics = _evaluate_subject_level(model, val_loader, device)
            mean_loss = running["loss"] / max(running["steps"], 1)
            lr = float(optimizer.param_groups[0]["lr"])

            record = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_ssl": running["ssl"] / max(running["steps"], 1),
                "train_t1": running["t1"] / max(running["steps"], 1),
                "train_t2": running["t2"] / max(running["steps"], 1),
                "val_mean_acc": val_metrics["mean_acc"],
                "val_left_acc": val_metrics["left_acc"],
                "val_right_acc": val_metrics["right_acc"],
                "lr": lr,
            }
            history.append(record)

            logging.info(
                "Fold %d Epoch %d | loss=%.4f | val_mean_acc=%.4f (L=%.4f, R=%.4f)",
                fold_idx,
                epoch,
                record["train_loss"],
                record["val_mean_acc"],
                record["val_left_acc"],
                record["val_right_acc"],
            )

            if val_metrics["mean_acc"] > best_acc:
                best_acc = val_metrics["mean_acc"]
                best_state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                }
                torch.save(best_state, os.path.join(fold_dir, "best.pt"))

        with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        fold_summary = {
            "fold": fold_idx,
            "best_acc": best_acc,
            "best_epoch": int(best_state["epoch"]) if best_state else -1,
            "checkpoint": os.path.join(fold_dir, "best.pt"),
        }
        fold_summaries.append(fold_summary)

    cv_result = {
        "num_folds": cv_folds,
        "folds": fold_summaries,
        "mean_best_acc": float(np.mean([f["best_acc"] for f in fold_summaries])) if fold_summaries else 0.0,
    }

    with open(os.path.join(work_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_result, f, indent=2)

    logging.info("CV done: %s", cv_result)
    return cv_result
