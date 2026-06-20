import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED,
    MODEL_DIR, PLOT_DIR, RESULT_DIR
)
from src.model import FakeNewsDetector
from src.preprocessing import FakeNewsDataset

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def train_model(df, source_features, sentiment_features):
    texts = df["combined_text"].tolist()
    labels = df["label"].tolist()

    # Train / Val / Test split (70/15/15)
    (X_train, X_temp, src_train, src_temp,
     sent_train, sent_temp, y_train, y_temp) = train_test_split(
        texts, source_features, sentiment_features, labels,
        test_size=0.30, random_state=SEED, stratify=labels
    )

    (X_val, X_test, src_val, src_test,
     sent_val, sent_test, y_val, y_test) = train_test_split(
        X_temp, src_temp, sent_temp, y_temp,
        test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_ds = FakeNewsDataset(X_train, src_train, sent_train, y_train)
    val_ds   = FakeNewsDataset(X_val,   src_val,   sent_val,   y_val)
    test_ds  = FakeNewsDataset(X_test,  src_test,  sent_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = FakeNewsDetector().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            optimizer.zero_grad()

            input_ids       = batch["input_ids"].to(DEVICE)
            attention_mask  = batch["attention_mask"].to(DEVICE)
            source_feats    = batch["source_features"].to(DEVICE)
            sentiment_feats = batch["sentiment_features"].to(DEVICE)
            labels_b        = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, source_feats, sentiment_feats)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels_b.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_b).sum().item()
            total += labels_b.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ---- VALIDATION ----
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                input_ids       = batch["input_ids"].to(DEVICE)
                attention_mask  = batch["attention_mask"].to(DEVICE)
                source_feats    = batch["source_features"].to(DEVICE)
                sentiment_feats = batch["sentiment_features"].to(DEVICE)
                labels_b        = batch["label"].to(DEVICE)

                logits = model(input_ids, attention_mask, source_feats, sentiment_feats)
                loss = criterion(logits, labels_b)

                v_loss += loss.item() * labels_b.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels_b).sum().item()
                v_total += labels_b.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            print(f"  --> Best model saved at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pt"))

    # Plot curves
    plot_training_curves(history)

    return model, test_loader, X_test, y_test, src_test, sent_test


def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved.")