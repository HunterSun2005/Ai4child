# Kaggle Track12 Guide (EfficientGCNv1)

This document is the dedicated runbook for the Kaggle competition pipeline in this repo.
It supports **Track1 + Track2 joint training/inference** with a shared backbone and independent task heads.

## 1. Expected File Layout

Place all competition files inside `EfficientGCNv1`:

- `./dataset/` (full pose dataset)
- `./train/track1_train.json`
- `./train/track2_train.json`
- `./submissions/submission_template.csv` (template)

Default config already points to this layout:

- `configs/track12_multitask_b0.yaml`

## 2. Entry Script

Use:

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml <command>
```

Supported commands:

- `preprocess`: build clip cache + manifest
- `train`: run CV training
- `predict`: run model inference (track1/track2/both)
- `make_submission`: render final CSV from prediction JSON

## 3. Run Commands

### 3.1 Preprocess

```bash
python track12_main.py preprocess
```

Optional smoke test:

```bash
python track12_main.py preprocess --max_clips 20
```

### 3.2 Train (5-fold by default)

```bash
python track12_main.py train --cv 5
```

### 3.3 Predict

Predict both tracks with all available folds:

```bash
python track12_main.py predict --folds all --task both
```

Task options:

- `--task track1`
- `--task track2`
- `--task both`

### 3.4 Make Submission

```bash
python track12_main.py make_submission
```

Default output:

- `./submissions/submission_track12.csv`

## 4. What Gets Filled in Submission

- `track1-*` rows:
  - `L1..L17`, `R1..R17`, `Total`
- `track2-*` rows:
  - `Left_gait_subtype`, `Right_gait_subtype`

Unnamed trailing columns in templates are dropped automatically to avoid Kaggle `null values` errors.

## 5. Key Paths in Config

From `configs/track12_multitask_b0.yaml`:

- `paths.dataset_root: ./dataset`
- `paths.track1_label: ./train/track1_train.json`
- `paths.track2_label: ./train/track2_train.json`
- `paths.submission_template: ./submissions/submission_template.csv`
- `paths.work_dir: ./workdir/aichild_track12`
- `paths.submission_output: ./submissions/submission_track12.csv`

## 6. Troubleshooting

- `Prediction requires PyTorch`
  - Install PyTorch in your runtime environment first.
- `No fold checkpoints found`
  - Run training first, or point `--folds` to existing fold ids.
- Kaggle says `Submission contains null values`
  - Re-run `make_submission` with the latest code; unnamed columns are auto-cleaned.

