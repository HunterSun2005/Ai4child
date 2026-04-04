# Kaggle Track12 Guide (EfficientGCNv1)

This document is the dedicated runbook for the Kaggle competition pipeline in this repo.
It supports **Track1 + Track2 joint training/inference** with a shared backbone and independent task heads.

Default keypoint setting uses **65 non-face points** from COCO-WholeBody:

- body + feet: `0..22`
- both hands: `91..132`
- face (`23..90`) is excluded

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

Optional experiment directory override:

```bash
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 <command>
```

With `--work_dir`, the pipeline automatically redirects:

- `paths.work_dir`
- `paths.cache_dir`
- `paths.manifest_path`
- `paths.pca_model_path`
- `paths.prediction_path`

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

PCA ablation example:

```bash
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 preprocess --use_pca on --pca_components 32
```

When PCA is enabled, preprocessing also fits and saves a PCA model.

### 3.2 Train (3-fold by default)

```bash
python track12_main.py train --cv 3
```

PCA ablation example:

```bash
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 train --cv 3 --use_pca on --pca_components 32
```

Each fold now saves **two task-specific best checkpoints**:

- `best_track1.pt`: selected by Track1 validation metric (`mean_f1`)
- `best_track2.pt`: selected by Track2 validation metric (`mean_acc`)
- `best.pt`: compatibility alias (same as `best_track2.pt`)

### 3.3 Predict

Predict both tracks with all available folds:

```bash
python track12_main.py predict --folds all --task both
```

Optional: only ensemble top-k folds by validation metric from `cv_summary.json`:

```bash
python track12_main.py predict --folds all --task both --ensemble_topk 2
```

If training used PCA, prediction should use the same PCA setting:

```bash
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 predict --folds all --task both --use_pca on --pca_components 32
```

Default checkpoint policy is `separate`:

- Track1 inference uses `best_track1.pt`
- Track2 inference uses `best_track2.pt`

You can switch policy explicitly:

```bash
# recommended: task-specific selection
python track12_main.py predict --folds all --task both --checkpoint_policy separate

# legacy behavior: both tasks use track2-selected checkpoints
python track12_main.py predict --folds all --task both --checkpoint_policy shared
```

Task options:

- `--task track1`
- `--task track2`
- `--task both`
- `--ensemble_topk K` (e.g. `2`), choose top-k folds by CV metric instead of averaging all folds

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
- `paths.pca_model_path: ./workdir/aichild_track12/pca_joint_model.npz`
- `paths.submission_template: ./submissions/submission_template.csv`
- `paths.work_dir: ./workdir/aichild_track12`
- `paths.submission_output: ./submissions/submission_track12.csv`

PCA config:

- `data.pca.enabled`: whether to enable PCA reduction+reconstruction (keeps original graph/keypoint size for GCN compatibility)
- `data.pca.n_components`: PCA latent dimension (for 65 keypoints, common trials: `16/24/32/48`)
- `data.pca.fit_on`: `non_test` / `labeled` / `all`
- `data.pca.frames_per_clip`: sampled frames per clip used to fit PCA

## 6. Troubleshooting

- `Prediction requires PyTorch`
  - Install PyTorch in your runtime environment first.
- `No fold checkpoints found`
  - Run training first, or point `--folds` to existing fold ids.
- Kaggle says `Submission contains null values`
  - Re-run `make_submission` with the latest code; unnamed columns are auto-cleaned.
