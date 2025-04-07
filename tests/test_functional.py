import pytest
import pandas as pd

# 功能测试：从文本进行预测
def test_functional_from_text(predictor):
    text = "This is a news article about sports."
    label = predictor.predict(text)
    assert label == "Sports"  # 假设预测为 Sports 类别

def test_predict_from_csv(predictor):
    test_data = pd.read_csv("dataset/test_data.csv")
    for _, row in test_data.iterrows():
        text = row['text']
        expected_label = row['label']
        label = predictor.predict(text)
        assert label == expected_label
