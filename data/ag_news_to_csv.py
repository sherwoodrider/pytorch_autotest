# scripts/ag_news_to_csv.py
from datasets import load_dataset
import pandas as pd
import os

def download_and_save_agnews(save_path="data/ag_news.csv", split="train", limit=None):
    print(f"🔽 下载 AG News 数据集 ({split})...")
    dataset = load_dataset("ag_news", split=split)

    print(f"📄 数据集大小: {len(dataset)} 条")

    if limit:
        dataset = dataset.select(range(limit))
        print(f"✂️ 截取前 {limit} 条记录")

    # 将标签索引映射为类别文本
    label_map = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    texts = dataset["text"]
    labels = [label_map[lbl] for lbl in dataset["label"]]

    df = pd.DataFrame({"text": texts, "label": labels})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ 数据保存至: {save_path} (共 {len(df)} 条)")

if __name__ == "__main__":
    download_and_save_agnews(split="train", save_path="data/ag_news_train.csv", limit=5000)
    download_and_save_agnews(split="test", save_path="data/ag_news_test.csv", limit=1000)
