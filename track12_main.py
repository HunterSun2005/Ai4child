import argparse
import logging
import os
from typing import Any, Dict

import yaml


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[ %(asctime)s ] %(message)s",
    )


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    return cfg


def _apply_runtime_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    paths_cfg = config.setdefault("paths", {})
    data_cfg = config.setdefault("data", {})
    pca_cfg = data_cfg.setdefault("pca", {})

    # Optional experiment directory override.
    # When provided:
    # - cache_dir is shared in "<parent_of_work_dir>/cache"
    # - other artifacts stay isolated in the work_dir itself
    work_dir_override = str(getattr(args, "work_dir", "") or "").strip()
    if work_dir_override:
        work_dir_norm = os.path.normpath(work_dir_override)
        default_parent = os.path.dirname(os.path.normpath(paths_cfg.get("work_dir", "./workdir")))
        parent_dir = os.path.dirname(work_dir_norm) or default_parent or "."
        paths_cfg["work_dir"] = work_dir_norm
        paths_cfg["cache_dir"] = os.path.join(parent_dir, "cache")
        paths_cfg["manifest_path"] = os.path.join(work_dir_norm, "manifest.csv")
        paths_cfg["pca_model_path"] = os.path.join(work_dir_norm, "pca_joint_model.npz")
        paths_cfg["prediction_path"] = os.path.join(work_dir_norm, "predictions_multitask.json")

    use_pca = getattr(args, "use_pca", "auto")
    if use_pca == "on":
        pca_cfg["enabled"] = True
    elif use_pca == "off":
        pca_cfg["enabled"] = False

    pca_components = int(getattr(args, "pca_components", -1))
    if pca_components > 0:
        pca_cfg["n_components"] = pca_components

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Track12 multitask pipeline for EfficientGCNv1")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/track12_multitask_b0.yaml",
        help="Path to yaml config",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="",
        help=(
            "Optional experiment directory override. "
            "Keeps cache in sibling shared folder '<parent>/cache' and stores other artifacts in this folder."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_pre = subparsers.add_parser("preprocess", help="Build cache and manifest")
    sp_pre.add_argument("--max_clips", type=int, default=-1, help="Only preprocess first N clips for smoke test")
    sp_pre.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    sp_pre.add_argument(
        "--use_pca",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Override PCA usage from config.",
    )
    sp_pre.add_argument(
        "--pca_components",
        type=int,
        default=-1,
        help="Override PCA components from config (effective when PCA enabled).",
    )

    sp_train = subparsers.add_parser("train", help="Run CV training")
    sp_train.add_argument("--cv", type=int, default=-1, help="Override CV folds; <=0 means use config")
    sp_train.add_argument("--epochs", type=int, default=-1, help="Override epochs in config")
    sp_train.add_argument(
        "--use_pca",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Override PCA usage from config.",
    )
    sp_train.add_argument(
        "--pca_components",
        type=int,
        default=-1,
        help="Override PCA components from config (effective when PCA enabled).",
    )

    sp_pred = subparsers.add_parser("predict", help="Predict track1/track2 test subjects")
    sp_pred.add_argument("--folds", type=str, default="", help="all or comma-separated fold ids, e.g. 1,2,3")
    sp_pred.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["track1", "track2", "both"],
        help="Prediction target task",
    )
    sp_pred.add_argument(
        "--checkpoint_policy",
        type=str,
        default=None,
        choices=["shared", "separate"],
        help=(
            "Checkpoint selection policy. "
            "'separate': track1 uses best_track1.pt and track2 uses best_track2.pt; "
            "'shared': both tasks use track2-selected checkpoints."
        ),
    )
    sp_pred.add_argument(
        "--use_pca",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Override PCA usage from config.",
    )
    sp_pred.add_argument(
        "--pca_components",
        type=int,
        default=-1,
        help="Override PCA components from config (effective when PCA enabled).",
    )
    sp_pred.add_argument(
        "--ensemble_topk",
        type=int,
        default=-1,
        help=(
            "Use top-k folds ranked by cv_summary metrics. "
            "<0 means use config, 0 means use all selected folds."
        ),
    )
    sp_pred.add_argument("--output", type=str, default="", help="Prediction output json path")

    sp_sub = subparsers.add_parser("make_submission", help="Fill submission template")
    sp_sub.add_argument("--predictions", type=str, default="", help="Prediction json path")
    sp_sub.add_argument("--template", type=str, default="", help="Submission template path")
    sp_sub.add_argument("--output", type=str, default="", help="Output submission csv path")

    args = parser.parse_args()
    _setup_logging()

    config_path = os.path.abspath(args.config)
    config = _load_config(config_path)
    config = _apply_runtime_overrides(config, args)

    if args.command == "preprocess":
        from src.aichild.data import preprocess_dataset

        info = preprocess_dataset(config, max_clips=args.max_clips, overwrite=args.overwrite)
        logging.info("Preprocess summary: %s", info)

    elif args.command == "train":
        try:
            from src.aichild.trainer import train_cv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Training requires PyTorch. Please install torch first."
            ) from exc

        cv_folds = int(args.cv)
        if cv_folds <= 0:
            cv_folds = int(config.get("train", {}).get("cv_folds", 3))
        result = train_cv(config, cv_folds=cv_folds, max_epochs=args.epochs)
        logging.info("Train summary: %s", result)

    elif args.command == "predict":
        try:
            from src.aichild.inference import predict_multitask
        except (ModuleNotFoundError, ImportError) as exc:
            raise ModuleNotFoundError(
                "Prediction requires PyTorch. Please install torch first."
            ) from exc

        try:
            pred_cfg = config.get("predict", {})
            folds = args.folds.strip() if isinstance(args.folds, str) else ""
            if not folds:
                folds = str(pred_cfg.get("folds", "all"))

            task = args.task.strip() if isinstance(args.task, str) else ""
            if not task:
                task = str(pred_cfg.get("task", "both"))

            checkpoint_policy = (
                args.checkpoint_policy.strip()
                if isinstance(args.checkpoint_policy, str)
                else ""
            )
            if not checkpoint_policy:
                checkpoint_policy = str(pred_cfg.get("checkpoint_policy", "separate"))

            ensemble_topk = int(args.ensemble_topk)
            if ensemble_topk < 0:
                ensemble_topk = int(pred_cfg.get("ensemble_topk", 0))

            predictions = predict_multitask(
                config,
                folds=folds,
                output_path=args.output,
                task=task,
                checkpoint_policy=checkpoint_policy,
                ensemble_topk=ensemble_topk,
            )
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Prediction requires PyTorch. Please install torch first."
            ) from exc
        logging.info(
            "Predicted subjects | track1=%d | track2=%d",
            len(predictions.get("track1", {})),
            len(predictions.get("track2", {})),
        )

    elif args.command == "make_submission":
        from src.aichild.inference import make_submission_from_template

        pred_path = args.predictions or config["paths"]["prediction_path"]
        template = args.template or config["paths"]["submission_template"]
        output = args.output or config["paths"]["submission_output"]
        out = make_submission_from_template(template, pred_path, output)
        logging.info("Submission file ready: %s", out)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
