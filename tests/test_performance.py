import time
import pytest

from src.utils.dir_utils import print_case_name


@print_case_name
def test_inference_time(predictor):
    text = "This is a normal text."
    start_time = time.time()
    predictor.predict(text)
    end_time = time.time()
    inference_time = end_time - start_time
    predictor.log_info(
        "start_time: {},end_time: {}, inference_time: {}".format(str(start_time), str(end_time),  str(inference_time)))
    assert inference_time < 2.0  # 小于 2 秒
