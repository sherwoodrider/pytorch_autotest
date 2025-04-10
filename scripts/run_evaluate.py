from src.model.evaluate import EvaluateModel
from src.model.predict import Predictor
import datetime
import os
import pandas as pd
from src.utils.dir_utils import get_save_log_path

base_path = os.path.dirname(os.getcwd())
log_folder = get_save_log_path(base_path)
evaluate = EvaluateModel(model_path="D:/hugging_face/models/bert-base-uncased",
                      checkpoint_path="../checkpoints/bert_epoch1.pt",save_log_path = log_folder)


data_path="../dataset/data/ag_news_test.csv"
evaluate.load_dataset(data_path)
evaluate.evaluate_model()
print(f"acc score:{evaluate.get_acc()}")


