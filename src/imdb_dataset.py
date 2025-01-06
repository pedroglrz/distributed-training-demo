from datasets import load_dataset
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class IMDBDataset(Dataset):
    def __init__(self, split="train", max_length=512, subset_size=1000, model_name="bert-base-uncased",logger = None):
        logger.log(f"[Process {os.getpid()}] Initializing IMDBDataset")

        # Load data
        dataset = load_dataset("imdb")[split]

        # subset data
        self.texts = dataset["text"][:subset_size]
        self.labels = dataset["label"][:subset_size]
        
        logger.log(f"[Process {os.getpid()}] Loaded {len(self.texts)} reviews")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess(self, text):
        # logger.log(f"[Process {os.getpid()}] Preprocessing review (first 50 chars): {text[:50]}...")
        
        # Tokenize and encode
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # logger.log(f"Process {os.getpid()} accessing index {idx}...")
        text = self.texts[idx]
        label = self.labels[idx]

        processed = self.preprocess(text)
        return {
            'input_ids': processed['input_ids'],
            'attention_mask': processed['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }