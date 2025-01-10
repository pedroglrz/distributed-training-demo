import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import gc

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2, freeze_bert=True):
        super().__init__()
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True  # Enable gradient checkpointing
        
        # Load model with memory optimizations
        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,  # Use fp32 instead of fp16
        )
        
        # Free up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Initialize classifier
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        if freeze_bert:
            # Freeze transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False
            # Ensure classifier parameters are trainable
            for param in self.classifier.parameters():
                param.requires_grad = True
        
    def forward(self, input_ids, attention_mask):
        # Regular forward pass without gradient context manager
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        
        # Clear cache
        del outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return logits