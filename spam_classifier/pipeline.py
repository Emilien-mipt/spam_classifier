from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from spam_classifier.config.core import Config


# Создание пайплайна
def define_pipeline(config: Config):
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(
            C=config.model.params.C,
            max_iter=config.model.params.max_iter
        ))
    ])
    return model
