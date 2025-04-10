import datetime
import functools
import os
def get_save_log_path(test_path):
    now = datetime.datetime.now()
    # 格式化时间为文件名格式
    str_now = now.strftime('%Y_%m_%d_%H_%M_%S')
    log_folder_name = "pytorch_" + str_now
    save_log_folder = os.path.join(test_path, "logs")
    test_log_folder = os.path.join(save_log_folder, log_folder_name)
    if not os.path.exists(test_log_folder):
        os.mkdir(test_log_folder)
    return test_log_folder

def print_case_name(func):
    @functools.wraps(func)
    def wrapper(predictor, *args, **kwargs):
        predictor.log_info(f'enter {func.__name__}()')
        result = func(predictor, *args, **kwargs)
        predictor.log_info(f'quit {func.__name__}()')
        return result
    return wrapper