# run_pipeline.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import random
import pandas as pd
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from data_loader import load_dataset
from generate_html_report import generate_html_report
from nlpaug.augmenter.char import RandomCharAug
from nlpaug.augmenter.word import SynonymAug
from nlpaug.augmenter.word import ContextualWordEmbsAug


# 模型训练与评估
def train_and_evaluate():
    print("🔄 开始训练和评估模型...")

    # 加载数据
    train_texts, train_labels, test_texts, test_labels, label_encoder = load_dataset("data/ag_news_train.csv")

    # 训练模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_texts,
        eval_dataset=test_texts,
        tokenizer=model.config.tokenizer,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions, p.label_ids)},
    )
    trainer.train()

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"✅ 评估结果: {eval_results}")


# 对抗样本生成
def augment_and_generate_report():
    print("🔄 开始生成对抗样本并生成报告...")

    # 加载增强数据
    df = pd.read_csv("data/ag_news_test.csv")

    # 选择增强方法
    aug_mode = random.choice(["spelling", "synonym", "insert"])

    # 数据增强（拼写错误、同义词替换、插入）
    if aug_mode == "spelling":
        augmenter = RandomCharAug(action="swap")
    elif aug_mode == "synonym":
        augmenter = SynonymAug(aug_src='wordnet')
    else:
        augmenter = ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='insert')

    df["text_augmented"] = df["text"].apply(lambda x: augmenter.augment(x))

    # 伪造预测结果（这里可以用模型进行真实预测）
    df["predicted"] = df["label"].apply(lambda x: x if random.random() > 0.2 else random.choice(df["label"].unique()))
    df["correct"] = df["predicted"] == df["label"]

    # 生成 HTML 报告
    generate_html_report(df, output_path="reports/adversarial_report.html")

    print("✅ 对抗样本报告已生成，正在发送邮件...")


# 发送报告到指定邮箱
def send_email_with_report(recipient_email):
    print(f"📧 发送报告到 {recipient_email}...")

    # 邮件内容设置
    sender_email = "your_email@example.com"  # 用你的邮箱替换
    sender_password = "your_email_password"  # 用你的邮箱密码替换

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "对抗样本测试报告"

    body = "请查看附件中的对抗样本测试报告。"
    msg.attach(MIMEText(body, "plain"))

    # 附件
    with open("reports/adversarial_report.html", "r", encoding="utf-8") as f:
        report_content = f.read()

    msg.attach(MIMEText(report_content, "html"))

    try:
        # SMTP 服务（例如，使用 Gmail）
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("✅ 邮件已成功发送！")
    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")


# 主流程执行
if __name__ == "__main__":
    train_and_evaluate()  # 模型训练与评估
    augment_and_generate_report()  # 数据增强与报告生成
    send_email_with_report("recipient_email@example.com")  # 发送报告到邮箱
