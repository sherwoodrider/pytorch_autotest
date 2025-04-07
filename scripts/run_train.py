import datetime
from src.model.train import Trainer
from src.test_log.logger import TestLog
import os
def get_save_log_path(test_path):
    now = datetime.datetime.now()
    # 格式化时间为文件名格式
    str_now = now.strftime('%Y_%m_%d_%H_%M_%S')
    log_folder_name = "pytorch_" + str_now
    save_log_folder = os.path.join(test_path, "logs")
    test_log_folder = os.path.join(save_log_folder, log_folder_name)
    if not os.path.exists(test_log_folder):
        os.mkdir(test_log_folder)
    return test_log_folder

base_path = os.path.dirname(os.getcwd())
log_folder = get_save_log_path(base_path)
trainer = Trainer(model_path="D:/hugging_face/models/bert-base-uncased",
                  data_path="/dataset/data", save_log_path = log_folder)
trainer.train()
