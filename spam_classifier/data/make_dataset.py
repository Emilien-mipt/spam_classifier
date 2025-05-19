import pandas as pd
from sklearn.model_selection import train_test_split
from spam_classifier.data.preprocess import preprocess_text
import yaml
import os


def load_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config['data']['raw_data_path'], encoding='latin-1')
    data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Препроцессинг текста
    data['processed_text'] = data['text'].apply(preprocess_text)

    # Разделение данных
    train, test = train_test_split(
        data,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Сохранение данных
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    train.to_csv(os.path.join(config['data']['processed_path'], 'train.csv'), index=False)
    test.to_csv(os.path.join(config['data']['processed_path'], 'test.csv'), index=False)

    return train, test

if __name__ == '__main__':
    load_data("config/config.yaml")