import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from spam_classifier.config.paths import (CONFIG_FILE_PATH,
                                          PROCESSED_DATA_PATH,
                                          LOG_DIR,
                                          TRAINED_MODEL_DIR)
from spam_classifier.config.core import (create_and_validate_config,
                                         fetch_config_from_yaml)
from spam_classifier.pipeline import define_pipeline


def setup_logger(log_to_file: bool) -> logging.Logger:
    logger = logging.getLogger("spam_classifier.training")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(LOG_DIR / log_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_metrics(y_true, y_pred, metrics):
    results = {}
    for metric in metrics:
        if metric == "accuracy":
            results[metric] = accuracy_score(y_true, y_pred)
        elif metric == "precision":
            results[metric] = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            results[metric] = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "f1":
            results[metric] = f1_score(y_true, y_pred, zero_division=0)
    return results


def get_cv_scoring(metrics):
    scoring = {}
    for metric in metrics:
        if metric in {"accuracy", "precision", "recall", "f1"}:
            scoring[metric] = metric
    return scoring


def log_cv_results(logger, cv_results):
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        name = key.replace("test_", "")
        for idx, value in enumerate(values, start=1):
            logger.info("cv.fold_%d.%s=%.4f", idx, name, value)
        mean_val = values.mean()
        std_val = values.std()
        logger.info("cv.%s_mean=%.4f cv.%s_std=%.4f", name, mean_val, name, std_val)


def train_model(config_path):
    parsed_config = fetch_config_from_yaml(Path(config_path))
    config = create_and_validate_config(parsed_config)
    logger = setup_logger(config.training.log_to_file)

    # Загрузка данных
    train_data = pd.read_csv(PROCESSED_DATA_PATH.joinpath('train.csv'))
    logger.info("Train data loaded: %s rows", len(train_data))

    model = define_pipeline(config)

    # Кросс-валидация на обучающей выборке
    if config.training.run_validation:
        scoring = get_cv_scoring(config.training.metrics)
        if scoring:
            cv = StratifiedKFold(
                n_splits=config.training.cv_folds,
                shuffle=True,
                random_state=config.data.random_state,
            )
            cv_results = cross_validate(
                model,
                train_data["text"],
                train_data["label"],
                cv=cv,
                scoring=scoring,
                n_jobs=None,
            )
            log_cv_results(logger, cv_results)

    # Обучение финальной модели на всей обучающей выборке
    model.fit(train_data['text'], train_data['label'])
    logger.info("Training completed")

    # Оценка финальной модели на holdout-тесте
    if config.training.use_holdout:
        test_path = PROCESSED_DATA_PATH.joinpath('test.csv')
        if test_path.is_file():
            test_data = pd.read_csv(test_path)
            logger.info("Holdout test data loaded: %s rows", len(test_data))
            preds = model.predict(test_data['text'])
            metrics = compute_metrics(
                test_data['label'],
                preds,
                config.training.metrics,
            )
            for name, value in metrics.items():
                logger.info("holdout.%s=%.4f", name, value)
            report = classification_report(
                test_data['label'],
                preds,
                digits=4,
            )
            logger.info("classification_report:\n%s", report)
        else:
            logger.warning("Holdout evaluation skipped: test.csv not found at %s", test_path)

    # Сохранение модели
    if config.training.save_model:
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path_to_model = TRAINED_MODEL_DIR.joinpath(config.model.model_name)
        joblib.dump(model, path_to_model)
        logger.info("Model saved to %s", path_to_model)

    return model

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_model(sys.argv[1])
    else:
        train_model(CONFIG_FILE_PATH)
