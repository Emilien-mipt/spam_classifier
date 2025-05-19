from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Создание пайплайна
def define_pipeline(config):
    model = Pipeline([
        ('tfidf', CountVectorizer()),
        ('clf', LogisticRegression(
            C=config['model']['params']['C'],
            max_iter=config['model']['params']['max_iter']
        ))
    ])
    return model