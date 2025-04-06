import pandas as pd

# 构造合成数据
data = {
    "text": [
        "Hello world", "This is amazing", "You are fired!", "I love this product",
        "The economy is in decline", "Breaking: major earthquake hits Japan"
    ],
    "label": ["neutral", "positive", "negative", "positive", "negative", "world"]
}

df = pd.DataFrame(data)
df.to_csv("data/fake_dataset.csv", index=False)
