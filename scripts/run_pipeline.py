import pytest


def run_tests():
    pytest.main(["-v", "-k functional"])
    # pytest.main(["-v", "test_performance.py"])
    # pytest.main(["-v", "test_robustness.py"])
    # pytest.main(["-v", "test_security.py"])


def send_test_report_email(header, result_dict):
    pass


if __name__ == "__main__":
    result_dict = run_tests()
    # 构造测试完成的 header 和结果字典
    # header = "AGI自动化测试_2025-04-07"
    # send_test_report_email(header, result_dict)
