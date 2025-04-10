import datetime
from src.model.train import Trainer
from src.test_log.test_logger import TestLog
import os
from src.utils.dir_utils import get_save_log_path


base_path = os.path.dirname(os.getcwd())
log_folder = get_save_log_path(base_path)
trainer = Trainer(model_path="D:/hugging_face/models/bert-base-uncased",
                  data_path="../dataset/data", save_log_path = log_folder)
trainer.train()
