import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import os

# 定义数据增强器
spelling_aug = nac.RandomCharAug(action="swap")  # 随机交换字符（模拟拼写错误）
synonym_aug = naw.SynonymAug(aug_src='wordnet')  # 同义词替换（英文）
insert_aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='insert')

def augment_text(text, mode="spelling"):
    try:
        if mode == "spelling":
            return spelling_aug.augment(text)
        elif mode == "synonym":
            return synonym_aug.augment(text)
        elif mode == "insert":
            return insert_aug.augment(text)
        else:
            return text
    except Exception as e:
        print(f"[增强失败] {e}")
        return text

def augment_csv(input_path, output_path, mode="spelling", limit=1000):
    df = pd.read_csv(input_path)
    df = df.dropna().reset_index(drop=True)

    if limit:
        df = df.iloc[:limit]

    print(f"增强模式: {mode}，共处理 {len(df)} 条")

    df["text_augmented"] = df["text"].apply(lambda x: augment_text(x, mode))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"增强数据保存至: {output_path}")

if __name__ == "__main__":
    augment_csv(
        input_path="data/ag_news_test.csv",
        output_path="../dataset/data/ag_news_test_augmented_spelling.csv",
        mode="spelling",
        limit=500
    )
