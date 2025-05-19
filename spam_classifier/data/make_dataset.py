import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from spam_classifier.config.paths import (CONFIG_FILE_PATH,
                                          PROCESSED_DATA_PATH, RAW_DATA_PATH)
from spam_classifier.data.preprocess import preprocess_text


def load_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(RAW_DATA_PATH, encoding='iso-8859-1')
    data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    data.drop_duplicates(inplace=True)

    # Препроцессинг текста
    data['processed_text'] = data['text'].apply(preprocess_text)

    # Разделение данных
    train, test = train_test_split(
        data,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Сохранение данных
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DATA_PATH.joinpath('train.csv'), index=False)
    test.to_csv(PROCESSED_DATA_PATH.joinpath('test.csv'), index=False)

    return train, test

if __name__ == '__main__':
    if len(sys.argv) > 1:
        load_data(sys.argv[1])
    else:
        load_data(CONFIG_FILE_PATH)