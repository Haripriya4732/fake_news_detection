import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import (
    TRANSFORMER_MODEL, CREDIBILITY_DIM, SENTIMENT_DIM,
    TRANSFORMER_DIM, HIDDEN_DIM, DROPOUT, NUM_CLASSES
)


class FakeNewsDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(TRANSFORMER_MODEL)

        # Adaptive weighting (learnable weights for 3 streams)
        self.weight_text = nn.Parameter(torch.tensor(1.0))
        self.weight_source = nn.Parameter(torch.tensor(1.0))
        self.weight_sentiment = nn.Parameter(torch.tensor(1.0))

        # Feature projection layers
        fused_dim = TRANSFORMER_DIM + CREDIBILITY_DIM + SENTIMENT_DIM

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, source_features, sentiment_features):
        # Transformer CLS embedding
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # (B, 768)

        # Adaptive weighting
        w_text = torch.sigmoid(self.weight_text)
        w_source = torch.sigmoid(self.weight_source)
        w_sentiment = torch.sigmoid(self.weight_sentiment)

        weighted_text = w_text * cls_embedding
        weighted_source = w_source * source_features
        weighted_sentiment = w_sentiment * sentiment_features

        # Feature fusion (concatenation)
        fused = torch.cat([weighted_text, weighted_source, weighted_sentiment], dim=1)

        # Classification
        logits = self.classifier(fused)
        return logits