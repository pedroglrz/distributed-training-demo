"""IMDB dataset loader for distributed training demonstration.

This module provides a PyTorch Dataset implementation for loading and preprocessing
IMDB movie reviews for sentiment analysis in a distributed training setup.
"""

import gc
import logging
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class IMDBDataset(Dataset):
    """IMDB movie reviews dataset for sentiment analysis.
    
    This dataset loader is optimized for distributed training, ensuring efficient
    memory usage during data loading and preprocessing.
    
    Args:
        split: Dataset split to load ("train" or "test")
        max_length: Maximum sequence length for tokenization
        subset_size: Number of examples to load (for debugging/testing)
        model_name: Name of the pretrained model for tokenization
        verbose: Enable verbose logging
    """
    
    def __init__(
        self,
        split: str = "train",
        max_length: int = 256,
        subset_size: int = 256,
        model_name: str = "distilbert-base-uncased",
        verbose: bool = False
    ) -> None:
        logger.info(f"[Process {os.getpid()}] Initializing IMDBDataset")
        
        # Load data in smaller chunks
        dataset = load_dataset("imdb", split=f"{split}[:{subset_size}]")
        
        # Convert to list and clear cache
        self.texts: List[str] = dataset["text"]
        self.labels: List[int] = dataset["label"]
        self.verbose = verbose
        del dataset
        gc.collect()
        
        if self.verbose:
            logger.info(f"[Process {os.getpid()}] Loaded {len(self.texts)} reviews")
        
        # Initialize tokenizer with reduced memory footprint
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_length,
            padding_side='right',
            truncation_side='right',
        )
        self.max_length = max_length

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and preprocess a single text example.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None,  # Return python lists instead of tensors
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preprocessed example.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and label tensors
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        processed = self.preprocess(text)
        if self.verbose:
            logger.debug(f"Dataset accessing index: {idx}")
            
        return {
            'input_ids': processed['input_ids'],
            'attention_mask': processed['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }