from datasets import load_dataset
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import gc

class IMDBDataset(Dataset):
    def __init__(self, split="train", max_length=256, subset_size=256, model_name="distilbert-base-uncased",verbose = False):
        print(f"[Process {os.getpid()}] Initializing IMDBDataset")
        
        # Load data in smaller chunks
        dataset = load_dataset("imdb", split=f"{split}[:{subset_size}]")
        
        # Convert to list and clear cache
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.verbose = verbose
        del dataset
        gc.collect()
        
        if self.verbose:
            print(f"[Process {os.getpid()}] Loaded {len(self.texts)} reviews")
        
        # Initialize tokenizer with reduced memory footprint
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_length,
            padding_side='right',
            truncation_side='right',
        )
        self.max_length = max_length

    def preprocess(self, text):
        # More memory-efficient tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None,  # Return python lists instead of tensors
        )
        
        # Convert to tensors only when needed
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        processed = self.preprocess(text)
        # Add logging
        if self.verbose:
            print(f"Dataset accessing index: {idx}")
        return {
            'input_ids': processed['input_ids'],
            'attention_mask': processed['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }