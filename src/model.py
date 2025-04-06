import torch
import torch.nn as nn
import torch.nn.functional as F

class SpamClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.5):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # 输出 logits，交给 CrossEntropyLoss
        return x
