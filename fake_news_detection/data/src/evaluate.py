import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from torch.utils.data import DataLoader

from src.config import PLOT_DIR, RESULT_DIR, MODEL_DIR
from src.model import FakeNewsDetector
from src.preprocessing import FakeNewsDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids       = batch["input_ids"].to(DEVICE)
            attention_mask  = batch["attention_mask"].to(DEVICE)
            source_feats    = batch["source_features"].to(DEVICE)
            sentiment_feats = batch["sentiment_features"].to(DEVICE)
            labels_b        = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, source_feats, sentiment_feats)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=["Real (0)", "Fake (1)"]
    )
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds)

    # ROC Curve
    plot_roc_curve(all_labels, all_probs)

    # Precision-Recall Curve
    plot_precision_recall(all_labels, all_probs)

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved.")


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"ROC curve saved. AUC = {roc_auc:.4f}")


def plot_precision_recall(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="steelblue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print("Precision-Recall curve saved.")


def load_and_test(X_test, src_test, sent_test, y_test, model_path="best"):
    """Load saved model and run test evaluation."""
    model = FakeNewsDetector().to(DEVICE)

    fname = "best_model.pt" if model_path == "best" else "final_model.pt"
    path  = os.path.join(MODEL_DIR, fname)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"Model loaded from {path}")

    test_ds = FakeNewsDataset(X_test, src_test, sent_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=16)

    return evaluate_model(model, test_loader)