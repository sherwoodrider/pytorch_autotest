# visualization/gradio_app.py

import gradio as gr
from src.model.predict import Predictor
import datetime
from src.test_log.test_logger import TestLog
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
class GradioApp:
    def __init__(self, model_path, checkpoint_path):
        base_path = os.path.dirname(os.getcwd())
        log_folder = get_save_log_path(base_path)
        self.predictor = Predictor(model_path, checkpoint_path,save_log_path=log_folder)

    def predict_fn(self, text):
        label = self.predictor.predict(text)
        return f"预测结果：{label}"

    def launch(self):
        interface = gr.Interface(
            fn=self.predict_fn,
            inputs=gr.Textbox(lines=2, placeholder="请输入一段文本..."),
            outputs="text",
            title="文本分类 - BERT 模型",
            description="使用训练好的 BERT 模型进行文本分类预测"
        )
        interface.launch(server_port=7861,share=False)


if __name__ == "__main__":
    model_path = "D:/hugging_face/models/bert-base-uncased"
    checkpoint_path = "../checkpoints/bert_epoch1.pt"
    app = GradioApp(model_path, checkpoint_path)
    app.launch()
