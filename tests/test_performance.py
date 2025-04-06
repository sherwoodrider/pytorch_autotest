import time
import random
import pytest
import concurrent.futures
from src.train import train_model
from src.evaluate import predict_single_text

# 训练模型（用较少 epoch 减少测试时间）
model, dataset = train_model("data/spam.csv", epochs=3)
device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

# 测试用语料库
test_messages = [
    "Congratulations! You won a lottery of $10000!",
    "Hey, what's up? Let's meet tomorrow.",
    "Call this number now and win a free trip.",
    "Reminder: Your bill is due tomorrow.",
    "This is not spam, I swear.",
    "Urgent! Your account has been hacked!",
    "Can we reschedule our appointment?",
    "Make money fast using this secret method!",
] * 100  # 模拟大批量输入


# ---------- 1. 响应时间测试 ----------
def test_response_time():
    start = time.time()
    for msg in random.sample(test_messages, 100):
        predict_single_text(msg, model, dataset, device=device)
    end = time.time()
    avg_time = (end - start) / 100
    print(f"[Performance] Avg Response Time: {avg_time:.4f} sec")


# ---------- 2. 吞吐量测试（模拟并发） ----------
def test_throughput():
    def task(msg):
        return predict_single_text(msg, model, dataset, device=device)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(task, msg) for msg in test_messages[:200]]
        _ = [f.result() for f in futures]
    end = time.time()

    total_time = end - start
    throughput = 200 / total_time
    print(f"[Performance] Throughput: {throughput:.2f} samples/sec")


# ---------- 3. 长时间稳定性（小规模模拟） ----------
def test_stability():
    try:
        for i in range(10):
            for msg in test_messages[:50]:
                _ = predict_single_text(msg, model, dataset, device=device)
        print("[Performance] Stability test passed.")
    except Exception as e:
        pytest.fail(f"Model crashed during long run: {e}")
