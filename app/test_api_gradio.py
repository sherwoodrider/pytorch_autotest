# tests/test_api_gradio.py
import requests

def test_gradio_api():
    # 需要先运行 gradio_app.py
    text = "Congratulations! You've won $1000!"
    response = requests.post("http://127.0.0.1:7860/run/predict", json={
        "data": [text]
    })

    result = response.json()
    print(f"[Gradio API Test] => {result}")
