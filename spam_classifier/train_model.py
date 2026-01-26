import sys

import joblib
import pandas as pd
from pathlib import Path

from spam_classifier.config.paths import (CONFIG_FILE_PATH,
                                          PROCESSED_DATA_PATH,
                                          TRAINED_MODEL_DIR)
from spam_classifier.config.core import (create_and_validate_config,
                                         fetch_config_from_yaml)
from spam_classifier.pipeline import define_pipeline


def train_model(config_path):
    parsed_config = fetch_config_from_yaml(Path(config_path))
    config = create_and_validate_config(parsed_config)

    # Загрузка данных
    train_data = pd.read_csv(PROCESSED_DATA_PATH.joinpath('train.csv'))

    model = define_pipeline(config)

    # Обучение модели
    model.fit(train_data['text'], train_data['label'])

    # Сохранение модели
    if config.training.save_model:
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path_to_model = TRAINED_MODEL_DIR.joinpath(config.model.model_name)
        joblib.dump(model, path_to_model)

    return model

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_model(sys.argv[1])
    else:
        train_model(CONFIG_FILE_PATH)
