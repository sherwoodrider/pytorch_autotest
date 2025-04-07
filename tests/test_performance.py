import time
import pytest

# 性能测试：测量模型推理时间
def test_inference_time(predictor):
    text = "This is a normal text."
    start_time = time.time()
    predictor.predict(text)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 2.0  # 每个预测时间应小于 2 秒
