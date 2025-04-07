import os
from datetime import datetime
from sklearn.metrics import classification_report
from src.model.train import train_model
from src.evaluate import predict_single_text
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# ---------- 设置路径 ----------
os.makedirs("output", exist_ok=True)
img_path = "output/metrics_report.png"
report_path = "output/model_report.md"

# ---------- 加载模型 ----------
model, dataset = train_model("dataset/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

# ---------- 预测 ----------
test_samples = [
                   "Free entry in 2 a wkly comp to win FA Cup final tkts",
                   "Your appointment is confirmed for tomorrow",
                   "Congratulations! You’ve won a $500 gift card",
                   "Call me when you're free",
                   "URGENT! Your loan has been pre-approved!",
               ] * 20
true_labels = ["spam", "ham", "spam", "ham", "spam"] * 20
pred_labels = [predict_single_text(t, model, dataset, device=device) for t in test_samples]

# ---------- 分类指标 ----------
accuracy = accuracy_score(true_labels, pred_labels)
cls_report = classification_report(true_labels, pred_labels, output_dict=True)
f1_spam = cls_report["spam"]["f1-score"]
f1_ham = cls_report["ham"]["f1-score"]

# ---------- 混淆矩阵图 ----------
plt.figure(figsize=(6, 5))
cm = confusion_matrix(true_labels, pred_labels, labels=["spam", "ham"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["spam", "ham"], yticklabels=["spam", "ham"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(img_path)
plt.close()

# ---------- 响应时间 ----------
start = time.time()
for t in random.sample(test_samples, 10):
    predict_single_text(t, model, dataset, device=device)
end = time.time()
avg_response = (end - start) / 10

# ---------- 写入 Markdown ----------
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# 模型评估报告\n")
    f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## 1. 分类指标\n")
    f.write(f"- 准确率：**{accuracy:.2%}**\n")
    f.write(f"- Spam F1-score：**{f1_spam:.2f}**\n")
    f.write(f"- Ham F1-score：**{f1_ham:.2f}**\n\n")

    f.write("## 2. 性能指标\n")
    f.write(f"- 平均响应时间：**{avg_response:.4f} 秒/条**\n\n")

    f.write("## 3. 混淆矩阵图表\n")
    f.write(f"![混淆矩阵]({os.path.basename(img_path)})\n\n")

    f.write("## 4. 完整分类报告（JSON）\n")
    f.write("```json\n")
    import json

    f.write(json.dumps(cls_report, indent=4))
    f.write("\n```\n")

print(f"报告已生成：{report_path}")
