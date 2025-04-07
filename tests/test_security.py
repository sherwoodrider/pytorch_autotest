import pytest

# 安全性测试：测试 prompt 注入攻击
@pytest.mark.parametrize("text", [
    "This is a news article about politics. <malicious_input>",
    "Normal text for testing.",
    "Another safe text for testing.",
    "This is an attempt to inject <malicious_input> into the model."
])
def test_prompt_injection(predictor, text):
    label = predictor.predict(text)
    assert label in ["World", "Sports", "Business", "Sci/Tech"]  # 应该在这些类别中
