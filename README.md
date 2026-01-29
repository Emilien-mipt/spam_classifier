# spam_classifier

This project demonstrates how to package and train a simple spam/ham classifier with MLOps practices. It is designed for students learning how to structure ML code into modules, build training pipelines, configure via YAML, and add tests and CI.

## Project structure

- `spam_classifier/` — package code (pipeline, training, inference)
- `data/` — raw and processed datasets
- `config.yaml` — pipeline and training configuration
- `tests/` — pytest suite (unit + quality)
- `.github/workflows/ci.yml` — GitHub Actions CI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt -r requirements/requirements-dev.txt
```

## Data

Download and prepare the dataset:

```bash
make download_data
make process_data
```

`make process_data` builds `data/processed/train.csv` and `data/processed/test.csv`. The holdout split is controlled by:

- `data.test_size` in `config.yaml` (default 0.1)
- `training.use_holdout` (True/False)

## Training

Train with cross-validation and optional holdout evaluation:

```bash
make train
```

Training behavior is controlled in `config.yaml`:

- `training.cv_folds` — number of CV folds
- `training.metrics` — metrics to log (accuracy/precision/recall/f1/roc_auc)
- `training.use_holdout` — evaluate on `test.csv` if True
- `training.run_validation` — run CV if True

### Versioned artifacts

Package version is stored in `spam_classifier/_VERSION`. Model and log filenames include this version:

- Model: `spam_classifier/models/spam_classifier_vX.Y.Z.pkl`
- Logs: `spam_classifier/logs/logs_X.Y.Z.log`

## Inference

Single message:

```bash
python -m spam_classifier.predict "Free prize! Call now"
```

Batch inference from file (one message per line):

```bash
python -m spam_classifier.predict data/processed/test.csv -o results/preds.csv
```

Options:

- `-o/--output` — output CSV path (default: project root)
- `--no-message` — exclude message text from output CSV

## Tests

Run full test suite:

```bash
pytest tests
```

Quality tests (require trained model and holdout data):

```bash
pytest -m quality
```

## CI

GitHub Actions runs on PRs to `main` and `develop`:

- `black --check`
- `flake8`
- `mypy`
- `pytest tests`

## Pre-commit

Install and run pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

Hooks included: `black`, `flake8`, `mypy`.
