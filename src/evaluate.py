import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_dataset, batch_size=32, device="cpu"):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def predict_single_text(text, model, dataset_obj, device="cpu"):
    # 单条文本预测函数（附带标签编码器 + vectorizer 支持）
    model.eval()
    vectorizer = dataset_obj.vectorizer
    label_encoder = dataset_obj.label_encoder

    # 文本向量化
    x = vectorizer.transform([text]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        label = label_encoder.inverse_transform([predicted_class])[0]

    return label
