import pytest
from src.train import train_model
from src.evaluate import predict_single_text

# 准备模型与数据
model, dataset = train_model("data/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"


# ---------- 1. 正常输入 ----------
@pytest.mark.parametrize("text", [
    "You have won a free ticket! Click here!",
    "Hi, just checking in. Are you coming today?",
])
def test_valid_input(text):
    label = predict_single_text(text, model, dataset, device=device)
    assert label in ["spam", "ham"], f"Unexpected label: {label}"
    print(f"[Functional] Input: '{text}' => Label: {label}")


# ---------- 2. 空字符串输入 ----------
def test_empty_input():
    try:
        label = predict_single_text("", model, dataset, device=device)
        assert isinstance(label, str), "Output should be a string"
        print(f"[Functional] Empty input => Label: {label}")
    except Exception as e:
        pytest.fail(f"Model crashed on empty input: {e}")


# ---------- 3. 非法类型输入 ----------
@pytest.mark.parametrize("invalid_input", [None, 123, 45.6, [], {}])
def test_invalid_input_type(invalid_input):
    with pytest.raises(Exception):
        predict_single_text(invalid_input, model, dataset, device=device)
        print(f"[Functional] Invalid input: {invalid_input} => Should raise exception")


# ---------- 4. 一致性测试 ----------
def test_repeatability():
    text = "Free entry in a prize draw!"
    labels = [predict_single_text(text, model, dataset, device=device) for _ in range(5)]
    assert all(l == labels[0] for l in labels), f"Inconsistent predictions: {labels}"
    print(f"[Functional] Repeated prediction => {labels}")
