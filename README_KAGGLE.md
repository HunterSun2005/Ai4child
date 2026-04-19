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
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 -c configs/track12_multitask_b0.yaml <command>
```

With `--work_dir`, the pipeline will use:

- shared cache: `<parent_of_work_dir>/cache`
- isolated per-experiment files in `work_dir`: checkpoints, manifest, PCA model, predictions

Supported commands:

- `preprocess`: build clip cache + manifest
- `train`: run CV training
- `predict`: run model inference (track1/track2/both)
- `make_submission`: render final CSV from prediction JSON

## 3. Run Commands

Recommended minimal style (all behavior comes from config):

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml <command>
```

### 3.1 Preprocess

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml preprocess
```

Optional smoke test:

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml preprocess --max_clips 20
```

When PCA is enabled, preprocessing also fits and saves a PCA model.

### 3.2 Train (3-fold by default)

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml train
```

`cv` comes from `train.cv_folds` in config unless overridden.

Each fold now saves **two task-specific best checkpoints**:

- `best_track1.pt`: selected by Track1 validation metric (`mean_f1`)
- `best_track2.pt`: selected by Track2 validation metric (`mean_acc`)
- `best.pt`: compatibility alias (same as `best_track2.pt`)

### 3.3 Predict

Predict both tracks with all available folds:

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml predict
```

If `--folds/--task/--checkpoint_policy/--ensemble_topk` are omitted, it uses `predict.*` from config.

Optional: only ensemble top-k folds by validation metric from `cv_summary.json`:

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml predict --ensemble_topk 2
```

If training used PCA, prediction should use the same PCA setting:

```bash
python track12_main.py --work_dir ./workdir/aichild_track12_pca_32 -c configs/track12_multitask_b0.yaml predict
```

Default checkpoint policy is `separate`:

- Track1 inference uses `best_track1.pt`
- Track2 inference uses `best_track2.pt`

You can switch policy explicitly:

```bash
# recommended: task-specific selection
python track12_main.py -c configs/track12_multitask_b0.yaml predict --checkpoint_policy separate

# legacy behavior: both tasks use track2-selected checkpoints
python track12_main.py -c configs/track12_multitask_b0.yaml predict --checkpoint_policy shared
```

Task options:

- `--task track1`
- `--task track2`
- `--task both`
- `--ensemble_topk K` (e.g. `2`), choose top-k folds by CV metric instead of averaging all folds

### 3.4 Make Submission

```bash
python track12_main.py -c configs/track12_multitask_b0.yaml make_submission
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
- `paths.cache_dir: ./workdir/cache`
- `paths.pca_model_path: ./workdir/aichild_track12/pca_joint_model.npz`
- `paths.submission_template: ./submissions/submission_template.csv`
- `paths.work_dir: ./workdir/aichild_track12`
- `paths.submission_output: ./submissions/submission_track12.csv`

PCA config:

- `data.pca.enabled`: whether to enable PCA reduction+reconstruction (keeps original graph/keypoint size for GCN compatibility)
- `data.pca.n_components`: PCA latent dimension (for 65 keypoints, common trials: `16/24/32/48`)
- `data.pca.fit_on`: `non_test` / `labeled` / `all`
- `data.pca.frames_per_clip`: sampled frames per clip used to fit PCA

Fold config:

- `train.cv_folds`: training K-fold count (e.g. `1/3/5`)
- `predict.folds`: which fold checkpoints to use (`all` or `1,2,3`)

Score-aware graph config:

- `data.score.enabled`: enable confidence-aware graph aggregation
- `data.score.clip_min/clip_max`: normalize `keypoint_scores` to `[0,1]`
- `data.score.power`: confidence sharpening/smoothing exponent
- `data.score.only_above_thr`: set confidence to 0 for scores below `data.score_thr`

Track2-focused augmentation config (`train.augment`):

- Temporal: random crop-resample, shift, and speed perturbation
- Spatial: small rotation/translation/scale on joint coordinates
- Confidence: confidence noise + random/low-confidence dropout
- Left-right flip: horizontal flip with automatic `left/right` label swap and direction swap
- Rare-class boost: stronger augmentation for selected Track2 classes (default `type4/WNL`)

Sampler config:

- `train.use_track2_weighted_sampler`: enable class-aware weighted sampling
- `train.track2_class_weights`: class-level sampling weight map for Track2 labels

## 6. Troubleshooting

- `Prediction requires PyTorch`
  - Install PyTorch in your runtime environment first.
- `No fold checkpoints found`
  - Run training first, or point `--folds` to existing fold ids.
- Kaggle says `Submission contains null values`
  - Re-run `make_submission` with the latest code; unnamed columns are auto-cleaned.
