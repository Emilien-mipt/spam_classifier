import sys
from pathlib import Path
from typing import Iterable, Optional

import joblib
from pydantic import BaseModel, Field, ValidationError, field_validator

from spam_classifier.config.core import read_package_version
from spam_classifier.config.paths import TRAINED_MODEL_DIR


class PredictionInput(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def non_empty_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must be a non-empty string")
        return value


class PredictionOutput(BaseModel):
    label: str
    score: Optional[float] = None


def load_model():
    version = read_package_version()
    model_path = TRAINED_MODEL_DIR / f"spam_classifier_v{version}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path!s}. Train the model first."
        )
    return joblib.load(model_path)


def predict_message(message: str, model) -> PredictionOutput:
    validated = PredictionInput(message=message)
    pred = model.predict([validated.message])[0]
    label = "spam" if pred == 1 else "ham"

    score = None
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba([validated.message])[0][1])

    return PredictionOutput(label=label, score=score)


def predict_messages(messages: Iterable[str], model) -> list[PredictionOutput]:
    outputs: list[PredictionOutput] = []
    for message in messages:
        outputs.append(predict_message(message, model))
    return outputs


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m spam_classifier.predict \"your message\"")
        return 2

    model = load_model()
    arg = sys.argv[1]
    path = Path(arg)
    try:
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines()
            messages = [line for line in lines if line.strip()]
            outputs = predict_messages(messages, model)
            for idx, output in enumerate(outputs, start=1):
                if output.score is not None:
                    print(f"line={idx} label={output.label} score={output.score:.4f}")
                else:
                    print(f"line={idx} label={output.label}")
            return 0
        output = predict_message(arg, model)
    except (ValidationError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if output.score is not None:
        print(f"label={output.label} score={output.score:.4f}")
    else:
        print(f"label={output.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
