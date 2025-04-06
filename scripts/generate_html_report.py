# scripts/generate_html_report.py
import pandas as pd
from jinja2 import Template
import os

def generate_html_report(df, output_path="reports/adversarial_report.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    template = Template("""
    <html>
    <head>
        <meta charset="utf-8">
        <title>对抗样本测试报告</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f0f0f0; }
            .success { background-color: #e0ffe0; }
            .fail { background-color: #ffe0e0; }
        </style>
    </head>
    <body>
        <h1>🧪 对抗性鲁棒性测试报告</h1>
        <p>总样本数: {{ total }}，正确预测数: {{ correct }}，准确率: {{ acc }}%</p>
        <table>
            <tr>
                <th>#</th>
                <th>原始文本</th>
                <th>增强文本</th>
                <th>真实标签</th>
                <th>模型预测</th>
                <th>是否正确</th>
            </tr>
            {% for idx, row in df.iterrows() %}
            <tr class="{{ 'success' if row['correct'] else 'fail' }}">
                <td>{{ idx+1 }}</td>
                <td>{{ row['text'] }}</td>
                <td>{{ row['text_augmented'] }}</td>
                <td>{{ row['label'] }}</td>
                <td>{{ row['predicted'] }}</td>
                <td>{{ '✅' if row['correct'] else '❌' }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """)

    correct = (df["label"] == df["predicted"]).sum()
    total = len(df)
    acc = round(correct / total * 100, 2)

    html = template.render(df=df.iterrows(), correct=correct, total=total, acc=acc)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML 报告已生成: {output_path}")

# 示例调用
if __name__ == "__main__":
    df = pd.read_csv("data/ag_news_test_augmented_spelling.csv")

    # 伪造预测结果（你应替换为真实模型预测）
    import random
    df["predicted"] = df["label"].apply(lambda x: x if random.random() > 0.2 else random.choice(df["label"].unique()))
    df["correct"] = df["predicted"] == df["label"]

    generate_html_report(df)
