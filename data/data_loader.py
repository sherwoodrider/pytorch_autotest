# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

logger = get_logger()

def load_dataset(csv_path, text_col="text", label_col="label", test_size=0.2):
    logger.info(f"📂 加载数据集: {csv_path}")
    df = pd.read_csv(csv_path)

    # 清洗缺失值
    df = df[[text_col, label_col]].dropna()
    logger.info(f"✅ 数据集大小: {len(df)}")

    # 编码标签为数字
    label_encoder = LabelEncoder()
    df[label_col] = label_encoder.fit_transform(df[label_col])

    # 分割训练与测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_col], df[label_col], test_size=test_size, random_state=42
    )

    return (train_texts.tolist(), train_labels.tolist(),
            test_texts.tolist(), test_labels.tolist(),
            label_encoder)
