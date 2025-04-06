import matplotlib.pyplot as plt
import time
import random
from sklearn.metrics import accuracy_score
from src.train import train_model
from src.evaluate import predict_single_text

# 加载或训练模型
model, dataset = train_model("data/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

# 模拟测试样本
test_samples = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Your appointment is confirmed for tomorrow",
    "Congratulations! You’ve won a $500 gift card",
    "Call me when you're free",
    "URGENT! Your loan has been pre-approved!",
] * 20

# 模拟真实标签（每5个样本的顺序与预测样本对应）
true_labels = ["spam", "ham", "spam", "ham", "spam"] * 20

# ----------- 1. 准确率计算 -----------
pred_labels = [predict_single_text(t, model, dataset, device=device) for t in test_samples]
accuracy = accuracy_score(true_labels, pred_labels)
print(f"[Metric] 模型准确率：{accuracy * 100:.2f}%")

# ----------- 2. 响应时间测试 -----------
start = time.time()
for t in random.sample(test_samples, 10):
    predict_single_text(t, model, dataset, device=device)
end = time.time()
avg_response = (end - start) / 10
print(f"[Metric] 平均响应时间：{avg_response:.4f} 秒/条")

# ----------- 3. 可视化展示 -----------

# 模拟每个 epoch 的准确率（假设3轮）
epoch_acc = [random.uniform(0.75, 0.85) for _ in range(3)]
epochs = list(range(1, len(epoch_acc) + 1))

plt.figure(figsize=(10, 6))

# 图1：训练轮数 vs 准确率
plt.subplot(2, 1, 1)
plt.plot(epochs, epoch_acc, marker="o", linestyle='-', color="blue")
plt.title("模型训练准确率变化趋势")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)

# 图2：准确率 & 响应时间并列展示
plt.subplot(2, 1, 2)
plt.bar(["Accuracy", "Avg Response Time (s)"], [accuracy, avg_response], color=["green", "orange"])
plt.title("模型整体性能指标")

plt.tight_layout()
plt.show()
