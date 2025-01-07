import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2, freeze_bert=False):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        if freeze_bert:
            # Freeze all transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the last hidden state of the [CLS] token
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits