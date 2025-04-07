import pytest

# 鲁棒性测试：输入噪声数据进行预测
@pytest.mark.parametrize("text, expected_label", [
    ("!!! @@@ ##$$%%^^&&*()", "World"),
    ("", "World"),
    ("a" * 5000, "Sci/Tech"),
    ("Some random text with no meaning.", "Sports")
])
def test_predict_with_noisy_data(predictor, text, expected_label):
    label = predictor.predict(text)
    assert label == expected_label
