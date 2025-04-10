import logging
import os
import torch
from datasets import tqdm
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
from src.model.model_bert import BERTClassifier
from src.test_log.test_logger import TestLog
import pandas as pd

class EvaluateModel():
    def __init__(self, model_path, checkpoint_path,save_log_path,num_classes=4, device="cpu"):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BERTClassifier(pretrained_model_path=model_path, num_classes=num_classes)
        self.model.load(self.checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        # 标签映射（根据 ag_news 数据集）
        self.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        predictor_log_path = os.path.join(save_log_path, "evaluate.log")
        self.evaluate_logger = TestLog(log_file=predictor_log_path, level=logging.DEBUG)

        self.dataset = None
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def load_dataset(self,csv_path):
        self.dataset = pd.read_csv(csv_path)
    def evaluate_model(self):
        self.evaluate_logger.log_info("evaluate_model begin")
        if not self.dataset is None:
            texts = self.dataset["text"]
            all_preds = []
            all_labels = self.dataset["label"]
            count = 0
            length =  len(texts)
            with torch.no_grad():
                tqdm_texts = tqdm(texts, desc=f"loop {count + 1}/{length}", leave=False)
                for text in tqdm_texts:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        pred = torch.argmax(outputs, dim=1).item()

                    # pred_label = self.id2label.get(pred, "Unknown")
                    all_preds.append(pred)
                    self.evaluate_logger.log_info(
                        "input text: {}, output label: {}".format(text, pred))
            self.accuracy = accuracy_score(all_labels, all_preds)
            self.precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            self.recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            self.f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        else:
            self.evaluate_logger.log_error("self.dataset == None")
        self.evaluate_logger.log_info(f"self.accuracy : {self.accuracy} ,self.precision : {self.precision} ,self.recall :{self.recall} , self.f1: {self.f1}")
        self.evaluate_logger.log_info("evaluate_model end")
    def get_acc(self):
        return self.accuracy
    def get_precision(self):
        return self.precision
    def get_recall(self):
        return self.recall
    def get_f1(self):
        return self.f1
