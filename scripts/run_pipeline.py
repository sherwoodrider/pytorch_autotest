import pytest


def run_tests():
    return pytest.main(["-s", "-v", "../tests"])
    # pytest.main(["-s", "-k functional","../tests"])# performance  robustness security


def send_test_report_email(header, result_dict):
    pass


if __name__ == "__main__":
    result_dict = run_tests()
    # 构造测试完成的 header 和结果字典
    # header = "AGI自动化测试_2025-04-07"
    # send_test_report_email(header, result_dict)
