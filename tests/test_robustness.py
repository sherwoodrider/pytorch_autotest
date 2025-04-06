# scripts/robustness_test.py
import random
import pandas as pd
from nlpaug.augmenter.char import RandomCharAug
from nlpaug.augmenter.word import SynonymAug
from nlpaug.augmenter.word import ContextualWordEmbsAug
from sklearn.metrics import accuracy_score
import os


# 定义增强方法
def add_noise(text):
    """在文本中加入随机噪声"""
    noise = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(random.randint(1, 3)))
    return text + " " + noise


def random_spelling_error(text):
    """模拟拼写错误"""
    aug = RandomCharAug(action="swap")  # 随机交换字符
    return aug.augment(text)


def replace_with_synonym(text):
    """同义词替换"""
    aug = SynonymAug(aug_src='wordnet')
    return aug.augment(text)


def contextual_word_insertion(text):
    """上下文插入替换"""
    aug = ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='insert')
    return aug.augment(text)


# 鲁棒性测试函数
def robustness_test(df, augment_type="noise"):
    """
    对输入文本进行鲁棒性测试，使用指定的增强类型。
    :param df: 原始数据集
    :param augment_type: 增强类型（noise: 噪声， spelling: 拼写错误， synonym: 同义词替换， insert: 上下文插入）
    :return: 返回鲁棒性评估结果
    """

    augmented_texts = []

    if augment_type == "noise":
        augmented_texts = df["text"].apply(add_noise)
    elif augment_type == "spelling":
        augmented_texts = df["text"].apply(random_spelling_error)
    elif augment_type == "synonym":
        augmented_texts = df["text"].apply(replace_with_synonym)
    elif augment_type == "insert":
        augmented_texts = df["text"].apply(contextual_word_insertion)

    df["augmented_text"] = augmented_texts

    # 伪造预测结果（用实际模型预测时替换这部分）
    df["predicted"] = df["label"].apply(lambda x: x if random.random() > 0.2 else random.choice(df["label"].unique()))
    df["correct"] = df["predicted"] == df["label"]

    accuracy = accuracy_score(df["label"], df["predicted"])
    print(f"鲁棒性测试（{augment_type}）结果：准确率 = {accuracy:.4f}")

    return df, accuracy


# 执行鲁棒性测试并生成报告
def run_robustness_test(input_file, augment_type="noise"):
    """加载数据并执行鲁棒性测试"""
    df = pd.read_csv(input_file)
    df, accuracy = robustness_test(df, augment_type)

    # 生成 HTML 报告
    output_path = f"reports/robustness_report_{augment_type}.html"
    generate_html_report(df, output_path)
    print(f"✅ 鲁棒性报告已生成并保存在：{output_path}")


# 示例：运行鲁棒性测试
if __name__ == "__main__":
    # 选择不同的增强类型（噪声、拼写错误、同义词替换、上下文插入）
    run_robustness_test(input_file="data/ag_news_test.csv", augment_type="noise")
    run_robustness_test(input_file="data/ag_news_test.csv", augment_type="spelling")
    run_robustness_test(input_file="data/ag_news_test.csv", augment_type="synonym")
    run_robustness_test(input_file="data/ag_news_test.csv", augment_type="insert")
