import os
import time
from datetime import datetime
from src.train import train_model
from src.evaluate import predict_single_text
from tests.test_functional import run_functional_tests
from tests.test_performance import run_performance_tests
from tests.test_adversarial import run_adversarial_tests
from visualize_metrics import generate_visualizations
from generate_report import generate_report

def pipeline():
    print("🚀 启动完整模型评估 Pipeline...\n")

    # Step 1: 模型训练
    print("✅ 步骤 1：加载数据并训练模型...")
    model, dataset = train_model("data/spam.csv", epochs=3)
    device = "cuda" if model.parameters().__next__().is_cuda else "cpu"

    # Step 2: 功能测试
    print("🔍 步骤 2：功能测试")
    run_functional_tests(model, dataset, device)

    # Step 3: 对抗性安全测试
    print("🛡️ 步骤 3：安全性 & 对抗测试")
    run_adversarial_tests(model, dataset, device)

    # Step 4: 性能测试
    print("⚙️ 步骤 4：性能与响应时间测试")
    avg_time = run_performance_tests(model, dataset, device)

    # Step 5: 评估指标 & 可视化图表
    print("📊 步骤 5：生成图表与性能可视化")
    acc, f1_dict = generate_visualizations(model, dataset, device)

    # Step 6: 生成 Markdown 报告
    print("📝 步骤 6：生成模型测试报告")
    generate_report(acc, f1_dict, avg_time)

    print("\n🎉 全流程结束！报告保存在 ./output/model_report.md")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    pipeline()

