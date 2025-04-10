import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_scheduler
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from src.model.model_bert import BERTClassifier
from src.test_log.test_logger import TestLog
from tqdm import tqdm

class Trainer:
    def __init__(self, model_path, data_path,save_log_path, batch_size=12, num_epochs=1, num_classes=4, device=None):
        self.model_path = model_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_log_path = save_log_path
        # 日志记录
        train_log_path = os.path.join(save_log_path,"train.log")
        self.logger = TestLog(log_file=train_log_path)
        # 加载数据
        self.dataset = self.load_data()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataset["train"]["label"])

        self.model = BERTClassifier(pretrained_model_path=model_path, num_classes=num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        # self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0,
        #                                   num_training_steps=len(self.dataset["train"]) * num_epochs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_data(self):
        dataset = load_dataset("csv", data_files={
            "train": os.path.join(self.data_path, "ag_news_train.csv"),
            "test": os.path.join(self.data_path, "ag_news_test.csv")
        })
        return dataset

    def tokenize(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    def encode_label(self, example):
        example["label"] = self.label_encoder.transform([example["label"]])[0]
        return example

    def preprocess_data(self):
        self.dataset = self.dataset.map(self.tokenize)
        self.dataset = self.dataset.map(self.encode_label)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def train(self):
        self.preprocess_data()
        train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset["test"], batch_size=self.batch_size)

        self.logger.log_info("Training started.")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)

            for batch in train_loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # if self.lr_scheduler:
                #     self.lr_scheduler.step()

                self.optimizer.zero_grad()
                total_loss += loss.item()

                train_loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            self.logger.log_info(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_loss:.4f}")
            self.evaluate(val_loader)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.log_info(f"Training completed in {duration:.2f} seconds.")
        self.save_model(0)

    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        self.logger.log_info(f"Validation Accuracy: {acc:.4f}")

    def save_model(self, epoch):
        model_path_out = f"../checkpoints/bert_epoch{epoch + 1}.pt"
        self.model.save(model_path_out)
        self.logger.log_info(f"Model saved to {model_path_out}")
