import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Инициализация компонентов NLP
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление пунктуации
    processed_text = re.sub(f'[{string.punctuation}]', '', text)

    return ' '.join(processed_text)