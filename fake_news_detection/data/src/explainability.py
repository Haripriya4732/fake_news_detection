import os
import torch
import numpy as np
import shap
import lime
import lime.lime_text
import matplotlib.pyplot as plt

from src.config import PLOT_DIR, MAX_LEN
from src.model import FakeNewsDetector
from src.preprocessing import tokenizer, clean_text
from src.source_credibility import get_source_credibility_features
from src.sentiment_module import get_sentiment_features
from src.config import MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PLOT_DIR, exist_ok=True)



def lime_explain(model, texts, source_domains, labels, n_samples=5):
    """Generate LIME explanations for n_samples examples."""
    model.eval()

    def predict_fn(text_list):
        """Prediction function for LIME — only text varies."""
        probs_list = []
        for txt in text_list:
            enc = tokenizer(
                clean_text(txt),
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            src_feat = torch.tensor(
                get_source_credibility_features("unknown"), dtype=torch.float
            ).unsqueeze(0).to(DEVICE)
            sent_feat = torch.tensor(
                get_sentiment_features(txt), dtype=torch.float
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                    src_feat,
                    sent_feat
                )
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list.append(probs)
        return np.array(probs_list)

    explainer = lime.lime_text.LimeTextExplainer(class_names=["Real", "Fake"])

    for i in range(min(n_samples, len(texts))):
        txt = texts[i]
        true_label = labels[i]

        exp = explainer.explain_instance(
            txt,
            predict_fn,
            num_features=15,
            num_samples=300
        )

        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME - Sample {i+1} | True: {'Fake' if true_label==1 else 'Real'}")
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, f"lime_sample_{i+1}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"LIME explanation saved: {save_path}")




def shap_explain(model, texts, source_domains, labels, n_samples=5):
    """Generate SHAP explanations using KernelExplainer."""
    model.eval()

    texts_clean = [clean_text(t) for t in texts[:n_samples]]

    def predict_fn_shap(text_list):
        probs_list = []
        for txt in text_list:
            enc = tokenizer(
                txt,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            src_feat = torch.tensor(
                get_source_credibility_features("unknown"), dtype=torch.float
            ).unsqueeze(0).to(DEVICE)
            sent_feat = torch.tensor(
                get_sentiment_features(txt), dtype=torch.float
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                    src_feat,
                    sent_feat
                )
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0][1]
            probs_list.append([1 - prob, prob])
        return np.array(probs_list)

    background = texts_clean[:min(3, len(texts_clean))]
    explainer  = shap.KernelExplainer(predict_fn_shap, background)

    shap_values = explainer.shap_values(texts_clean, nsamples=50)

    print("SHAP analysis complete. Saving summary plot...")
    plt.figure(figsize=(10, 4))
    shap.summary_plot(
        shap_values[1],
        features=np.array(texts_clean).reshape(-1, 1),
        feature_names=["text"],
        show=False,
        plot_type="bar"
    )
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "shap_summary.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"SHAP summary saved: {save_path}")