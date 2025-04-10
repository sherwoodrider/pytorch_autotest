import streamlit as st
import matplotlib.pyplot as plt
import random

from src.model.predict import Predictor
from sklearn.metrics import accuracy_score
import datetime
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



class StreamlitClass():
    def __init__(self):
        self.predictor = None
        self.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        self.texts = []
        self.predicted_labels = []
        self.true_labels = []

    def get_predictor(self):
        base_path = os.path.dirname(os.getcwd())
        log_folder = get_save_log_path(base_path)
        self.predictor = Predictor(model_path="D:/hugging_face/models/bert-base-uncased",
                                   checkpoint_path="../checkpoints/bert_epoch1.pt", save_log_path=log_folder)
    def predictor(self,text):
        label = self.predictor.predict(text)
        print(f"predictor success")
        return label
    def excute(self):
        input_text = st.text_area("输入短信内容进行预测", "请输入短信内容...")
        if input_text:
            if self.predictor == None:
                print("self.predictor == None")
            else:
                label = self.predictor(input_text)
                st.write(f"模型预测结果: {label}")
    def show_performance_plot(self):
        # 模拟准确率随时间变化（这里只是示意，实际可以更复杂）
        accuracies = [random.uniform(0.7, 0.9) for _ in range(10)]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), accuracies, marker="o", color="b", label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("模型准确率随训练进度变化")
        plt.legend(loc="lower right")
        st.pyplot(plt)

    def show_accuracy(self):
        # 模拟计算准确率
        accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        st.write(f"模型准确率：{accuracy * 100:.2f}%")


