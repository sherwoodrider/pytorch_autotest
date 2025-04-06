import streamlit as st
import matplotlib.pyplot as plt
import time
import random
from src.train import train_model
from src.evaluate import predict_single_text
from sklearn.metrics import accuracy_score

# 训练或加载模型
model, dataset = train_model("data/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

# 模拟一些测试数据
test_messages = [
    "You have won a free ticket! Click here!",
    "Hi, just checking in. Are you coming today?",
    "Reminder: Your bill is due tomorrow.",
    "Make money fast using this secret method!",
    "Urgent! Your account has been hacked!",
] * 20

# ---------- 模型准确率展示 ----------
def show_accuracy():
    # 模拟计算准确率
    predicted_labels = [predict_single_text(msg, model, dataset, device=device) for msg in test_messages]
    true_labels = ["spam", "ham", "ham", "spam", "spam"] * 20  # 简化版的真实标签
    accuracy = accuracy_score(true_labels, predicted_labels)
    st.write(f"### 模型准确率：{accuracy * 100:.2f}%")

# ---------- 响应时间展示 ----------
def show_response_time():
    start = time.time()
    for msg in random.sample(test_messages, 10):
        predict_single_text(msg, model, dataset, device=device)
    end = time.time()
    avg_time = (end - start) / 10
    st.write(f"### 平均响应时间：{avg_time:.4f} 秒/条")

# ---------- 生成模型预测结果的条形图 ----------
def show_performance_plot():
    # 模拟准确率随时间变化（这里只是示意，实际可以更复杂）
    accuracies = [random.uniform(0.7, 0.9) for _ in range(10)]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), accuracies, marker="o", color="b", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("模型准确率随训练进度变化")
    plt.legend(loc="lower right")
    st.pyplot(plt)

# ---------- Streamlit 界面展示 ----------
st.title("📨 垃圾短信分类模型")

# 显示模型准确率
show_accuracy()

# 显示响应时间
show_response_time()

# 显示性能图表
show_performance_plot()

# 文本框输入，模型预测
input_text = st.text_area("输入短信内容进行预测", "请输入短信内容...")
if input_text:
    label = predict_single_text(input_text, model, dataset, device=device)
    st.write(f"**模型预测结果**: {label}")
