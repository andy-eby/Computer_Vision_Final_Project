# Fine-Grained Bird Classification on CUB-200 (DINOv2 + Linear Probe)

This repository trains a linear probe on frozen DINOv2 image features for CUB-200-2011.

Current pipeline:
1. Extract and cache frozen DINOv2 features (`train`, `val`, `test`) as `.npz` files.
2. Train a scikit-learn pipeline: `StandardScaler -> LogisticRegression`.
3. Tune `C` with `GridSearchCV` + `StratifiedKFold`.
4. Evaluate Top-1 / Top-5 and generate analysis figures.

## Project Layout

- `datasets/cub200.py`: CUB parser using official train/test split and a stratified per-class validation carve-out from train.
- `datasets/transforms.py`: train/eval transforms.
- `models/dinov2.py`: frozen DINOv2 feature extractor.
- `scripts/extract_features.py`: standalone feature extraction and caching.
- `train.py`: end-to-end cache check + sklearn training + `results.json`.
- `evaluate.py`: post-training metrics and report figures.

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset Layout

Pass `--data_root` as the CUB folder containing:

- `images/`
- `images.txt`
- `image_class_labels.txt`
- `train_test_split.txt`
- `classes.txt`

Example:

```text
data/CUB_200_2011
```

## Quick Start

### 1) Train (end-to-end)

```bash
python train.py --data_root data/CUB_200_2011 --model_variant vitb14 --output_dir runs/cub_vitb14
```

What `train.py` does:
- ensures `train_features.npz`, `val_features.npz`, and `test_features.npz` exist in `--cache_dir` (extracts missing splits automatically),
- runs GridSearchCV over `classifier__C` with `StratifiedKFold`,
- reports test Top-1 and Top-5,
- prompts for confirmation if Top-1 is below 75%,
- saves:
  - `runs/.../sklearn_classifier.pkl`
  - `runs/.../results.json`

### 2) Evaluate + Generate Figures

```bash
python evaluate.py --data_root data/CUB_200_2011 --run_dir runs/cub_vitb14 --cache_dir cache
```

This computes Top-1/Top-5 from the saved sklearn model and writes figures to `runs/.../figures`.

## Script Usage

### `scripts/extract_features.py`

Extract one split at a time:

```bash
python scripts/extract_features.py --data_root data/CUB_200_2011 --split train --model_variant vitb14 --batch_size 128 --output_dir cache
python scripts/extract_features.py --data_root data/CUB_200_2011 --split val   --model_variant vitb14 --batch_size 128 --output_dir cache
python scripts/extract_features.py --data_root data/CUB_200_2011 --split test  --model_variant vitb14 --batch_size 128 --output_dir cache
```

Each output `.npz` contains:
- `features`: L2-normalized feature matrix
- `labels`: integer class IDs (0-based)

### `train.py`

Main flags:
- `--data_root` (required)
- `--model_variant` (`vits14`, `vitb14`, `vitl14`; default `vitb14`)
- `--output_dir` (default `./runs/cub_vitb14`)
- `--cache_dir` (default `./cache`)
- `--feature_batch_size` (default `128`)
- `--device` (`auto`, `cpu`, `cuda`; default `auto`)
- `--cv_folds` (default `5`)

Current `C` sweep in code:
- `[15.0, 17.5, 20.0, 22.5]`

### `evaluate.py`

Main flags:
- `--data_root` (required)
- `--run_dir` (required)
- `--cache_dir` (default `./cache`)
- `--model` (optional explicit model path)
- `--tsne_color_mode` (`class` or `order`, default `class`)
- `--order_map_json` (optional JSON map used when `--tsne_color_mode order`)
- `--tsne_samples` (default `1500`)
- `--seed` (default `42`)

Optional order-colored t-SNE:

```bash
python evaluate.py --data_root data/CUB_200_2011 --run_dir runs/cub_vitb14 --cache_dir cache --tsne_color_mode order --order_map_json order_map.json
```

## Outputs

### Cache directory

```text
cache/
  train_features.npz
  val_features.npz
  test_features.npz
```

### Run directory

```text
runs/cub_vitb14/
  sklearn_classifier.pkl
  results.json
  figures/
    confusion_matrix_full.npy
    confusions_top15_heatmap.png
    tsne_test_features.png
    per_class_lowest20.png
```

## Notes

- `results.json` is written by `train.py` and updated by `evaluate.py` with final evaluation fields.
