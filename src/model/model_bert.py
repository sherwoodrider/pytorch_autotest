import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(BERTClassifier, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        return self.classifier(output.pooler_output)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
