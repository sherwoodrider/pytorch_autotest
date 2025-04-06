import gradio as gr
from src.train import train_model
from src.evaluate import predict_single_text

# 训练或加载模型
model, dataset = train_model("data/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

def predict_gradio(text):
    label = predict_single_text(text, model, dataset, device=device)
    return f"模型预测：{label}"

# 启动 Gradio 界面
iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Textbox(lines=3, placeholder="请输入短信内容..."),
    outputs="text",
    title="Spam/Ham 消息分类模型",
    description="请输入一段短信内容，模型将判断是否为垃圾短信。"
)

if __name__ == "__main__":
    iface.launch()
