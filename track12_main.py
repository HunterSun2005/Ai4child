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


def main() -> None:
    parser = argparse.ArgumentParser(description="Track12 multitask pipeline for EfficientGCNv1")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/track12_multitask_b0.yaml",
        help="Path to yaml config",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_pre = subparsers.add_parser("preprocess", help="Build cache and manifest")
    sp_pre.add_argument("--max_clips", type=int, default=-1, help="Only preprocess first N clips for smoke test")
    sp_pre.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")

    sp_train = subparsers.add_parser("train", help="Run CV training")
    sp_train.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    sp_train.add_argument("--epochs", type=int, default=-1, help="Override epochs in config")

    sp_pred = subparsers.add_parser("predict", help="Predict track1/track2 test subjects")
    sp_pred.add_argument("--folds", type=str, default="all", help="all or comma-separated fold ids, e.g. 1,2,3")
    sp_pred.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["track1", "track2", "both"],
        help="Prediction target task",
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

        result = train_cv(config, cv_folds=args.cv, max_epochs=args.epochs)
        logging.info("Train summary: %s", result)

    elif args.command == "predict":
        try:
            from src.aichild.inference import predict_multitask
        except (ModuleNotFoundError, ImportError) as exc:
            raise ModuleNotFoundError(
                "Prediction requires PyTorch. Please install torch first."
            ) from exc

        try:
            predictions = predict_multitask(
                config,
                folds=args.folds,
                output_path=args.output,
                task=args.task,
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
