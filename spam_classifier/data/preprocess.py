import re
import string

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление пунктуации
    processed_text = re.sub(f'[{string.punctuation}]', '', text)

    return processed_text