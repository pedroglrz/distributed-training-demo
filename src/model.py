"""Transformer-based classifier for sentiment analysis.

This module implements a simple classifier that uses a pretrained transformer
model as the backbone, with an optional feature to freeze the transformer
weights during training.
"""

import gc
import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)

class TransformerClassifier(nn.Module):
    """Transformer-based classifier for sequence classification tasks.
    
    Args:
        model_name: Name of the pretrained transformer model to use
        num_classes: Number of output classes
        freeze_bert: Whether to freeze the transformer weights during training
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        freeze_bert: bool = True
    ) -> None:
        super().__init__()
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name)
        config.gradient_checkpointing = not freeze_bert
        
        # Load model with memory optimizations
        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
        )
        
        # Free up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize classifier
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        if freeze_bert:
            self._freeze_transformer()
            logger.info(
                "Parameter gradients status:\n"
                f"- Transformer requires_grad: {any(p.requires_grad for p in self.transformer.parameters())}\n"
                f"- Classifier requires_grad: {any(p.requires_grad for p in self.classifier.parameters())}"
            )
    
    def _freeze_transformer(self) -> None:
        """Freeze transformer parameters and ensure classifier remains trainable."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for padded sequences
            
        Returns:
            Logits for each class
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        
        # Clear cache
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return logits