UNAME_S := $(shell uname -s)

install:
	uv pip install -e ".[dev]"

download_data:
	mkdir -p data/raw
ifeq ($(UNAME_S),Linux)
	wget https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset -O data/raw/spam.zip
else
	curl -L -o data/raw/spam.zip \
	https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset
endif
	unzip data/raw/spam.zip -d data/raw/
	rm data/raw/spam.zip

process_data:
	uv run python -m spam_classifier.data.make_dataset config.yaml

train:
	uv run python -m spam_classifier.train_model config.yaml

quality:
	uv run pytest -m quality
