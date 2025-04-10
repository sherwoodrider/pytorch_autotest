from src.model.predict import Predictor
import datetime
import os
import pandas as pd
from src.utils.dir_utils import get_save_log_path

base_path = os.path.dirname(os.getcwd())
log_folder = get_save_log_path(base_path)
predictor = Predictor(model_path="D:/hugging_face/models/bert-base-uncased",
                      checkpoint_path="../checkpoints/bert_epoch1.pt",save_log_path = log_folder)


test_data_path = r"D:\code_repo\pytorch_autotest\dataset\data\ag_news_test.csv"
dataset = pd.read_csv(test_data_path)
texts = dataset["text"]
y_labels = dataset["label"]
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
count = 0
for t in texts:
    label = predictor.predict(t)
    print(f"the {str(count)} time predicted class: {label}")
    expected_label = id2label[int(y_labels[count])]
    # print(f"y_labels number : {y_labels[count]},label : {expected_label}")
    predictor.log_info("input text: {},expected label: {}, output label: {}".format(t, expected_label, label))
    count += 1
