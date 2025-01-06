import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2, freeze_bert=False):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        if freeze_bert:
            # Freeze all BERT parameters
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token for classification
        logits = self.classifier(pooled_output)
        return logits 