import logging
import os

import torch
from transformers import BertTokenizer
from src.model.model_bert import BERTClassifier
import sys

from src.test_log.logger import TestLog


class Predictor:
    def __init__(self, model_path, checkpoint_path,save_log_path, num_classes=4, device=None):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化 tokenizer、模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BERTClassifier(pretrained_model_path=model_path, num_classes=num_classes)
        self.model.load(self.checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        # 标签映射（根据 ag_news 数据集）
        self.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        predictor_log_path = os.path.join(save_log_path, "predictor.log")
        self.predictor_logger = TestLog(log_file=predictor_log_path, level=logging.DEBUG)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = torch.argmax(outputs, dim=1).item()

        label = self.id2label.get(pred, "Unknown")
        return label
