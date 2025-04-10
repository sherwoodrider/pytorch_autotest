import pytest

from src.model.predict import Predictor
from src.utils.dir_utils import get_save_log_path
import os

@pytest.fixture(scope="session")
def predictor():
    model_path = "D:/hugging_face/models/bert-base-uncased"
    checkpoint_path = "../checkpoints/bert_epoch1.pt"
    base_path = os.path.dirname(os.getcwd())
    log_folder = get_save_log_path(base_path)
    pre = Predictor(model_path=model_path, checkpoint_path=checkpoint_path,save_log_path=log_folder)
    yield pre