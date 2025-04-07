import pytest

from src.model.predict import Predictor

@pytest.fixture(scope="module")
def predictor():
    model_path = "D:/hugging_face/models/bert-base-uncased"
    checkpoint_path = "../checkpoints/bert_epoch1.pt"
    pre = Predictor(model_path=model_path, checkpoint_path=checkpoint_path)
    yield pre