import pandas as pd

# 创建功能测试数据
def generate_functionality_test_data():
    data = [
        {"text": "This is a news article about sports.", "label": "Sports"},
        {"text": "This is a news article about business.", "label": "Business"},
        {"text": "This is a news article about technology.", "label": "Sci/Tech"},
        {"text": "This is a news article about world events.", "label": "World"}
    ]
    df = pd.DataFrame(data)
    df.to_csv("dataset/functional_test_data.csv", index=False)
    print("Functionality test dataset generated successfully!")

# 创建性能测试数据
def generate_performance_test_data():
    data = [
        {"text": "Sample text for performance test."},
        {"text": "Another example text to test model performance."},
        {"text": "Quick brown fox jumps over the lazy dog."},
        {"text": "Test performance with a variety of different sentences."}
    ]
    df = pd.DataFrame(data)
    df.to_csv("dataset/performance_test_data.csv", index=False)
    print("Performance test dataset generated successfully!")

# 创建鲁棒性测试数据
def generate_robustness_test_data():
    data = [
        {"text": "!!! @@@ ##$$%%^^&&*()", "label": "World"},
        {"text": "", "label": "World"},  # 空文本
        {"text": "a" * 5000, "label": "Sci/Tech"},  # 超长文本
        {"text": "Some random text with no meaning.", "label": "Sports"}
    ]
    df = pd.DataFrame(data)
    df.to_csv("dataset/robustness_test_data.csv", index=False)
    print("Robustness test dataset generated successfully!")

# 创建安全性测试数据
def generate_security_test_data():
    data = [
        {"text": "This is a news article about politics. <malicious_input>", "label": "World"},
        {"text": "Normal text for testing.", "label": "Business"},
        {"text": "Another safe text for testing.", "label": "Sci/Tech"},
        {"text": "This is an attempt to inject <malicious_input> into the model.", "label": "World"}
    ]
    df = pd.DataFrame(data)
    df.to_csv("dataset/security_test_data.csv", index=False)
    print("Security test dataset generated successfully!")

# 主函数调用
if __name__ == "__main__":
    generate_functionality_test_data()
    generate_performance_test_data()
    generate_robustness_test_data()
    generate_security_test_data()
