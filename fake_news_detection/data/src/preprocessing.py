import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.config import TRANSFORMER_MODEL, MAX_LEN


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)


class FakeNewsDataset(Dataset):
    def __init__(self, texts, source_features, sentiment_features, labels):
        self.texts = [clean_text(t) for t in texts]
        self.source_features = source_features      # numpy array (N, CREDIBILITY_DIM)
        self.sentiment_features = sentiment_features  # numpy array (N, SENTIMENT_DIM)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "source_features": torch.tensor(self.source_features[idx], dtype=torch.float),
            "sentiment_features": torch.tensor(self.sentiment_features[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }