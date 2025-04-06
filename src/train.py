import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import SpamClassifier
from src.dataset import SpamDataset

def train_model(
    csv_path,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    hidden_dim=128,
    dropout=0.5,
    device=None,
    save_path=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 1. 加载数据
    dataset = SpamDataset(csv_path)
    train_ds, test_ds = dataset.preprocess()
    input_dim = dataset.get_input_dim()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型
    model = SpamClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. 开始训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 4. 保存模型（可选）
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[Info] Model saved to {save_path}")

    return model, dataset  # 返回模型和预处理器（用于后续评估）
