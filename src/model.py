import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import gc

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2, freeze_bert=True):
        super().__init__()
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name)
        # Only use gradient checkpointing if we're not freezing BERT
        config.gradient_checkpointing = not freeze_bert
        
        # Load model with memory optimizations
        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
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
                
            print("Parameter gradients status:")
            print(f"- Transformer requires_grad: {any(p.requires_grad for p in self.transformer.parameters())}")
            print(f"- Classifier requires_grad: {any(p.requires_grad for p in self.classifier.parameters())}")
        
    def forward(self, input_ids, attention_mask):
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