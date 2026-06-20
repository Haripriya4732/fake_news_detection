import os
import torch
import random
import numpy as np
from src.config import MODEL_DIR, MAX_LEN, SEED
from src.model import FakeNewsDetector
from src.preprocessing import tokenizer, clean_text
from src.source_credibility import get_source_credibility_features
from src.sentiment_module import get_sentiment_features

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model(model_path="best"):
    model = FakeNewsDetector().to(DEVICE)
    fname = "best_model.pt" if model_path == "best" else "final_model.pt"
    path  = os.path.join(MODEL_DIR, fname)

    if not os.path.exists(path):
        print(f"ERROR: Model not found at {path}")
        print("Please run main.py first to train and save the model.")
        exit()

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from: {path}\n")
    return model




def predict_single(model, text, source_domain="unknown"):
    clean  = clean_text(text)

    enc = tokenizer(
        clean,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    src_feat  = torch.tensor(
        get_source_credibility_features(source_domain), dtype=torch.float
    ).unsqueeze(0).to(DEVICE)

    sent_feat = torch.tensor(
        get_sentiment_features(clean), dtype=torch.float
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            src_feat,
            sent_feat
        )
        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label]) * 100

    return pred_label, confidence, probs


def print_result(text, source, pred_label, confidence, probs):
    label_str = "🔴 FAKE NEWS" if pred_label == 1 else "🟢 REAL NEWS"
   
    print(f"  Input Text   : {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"  Source       : {source}")
    print(f"  Prediction   : {label_str}")
    print(f"  Confidence   : {confidence:.2f}%")
    print(f"  Real Prob    : {probs[0]*100:.2f}%")
    print(f"  Fake Prob    : {probs[1]*100:.2f}%")
   




def test_random_samples(model, n=5):
    """Pick n random samples from the loaded dataset and predict."""
    from src.data_loader import load_all_datasets

    print(f"\nLoading dataset to pick {n} random samples...")
    df = load_all_datasets()
    samples = df.sample(n=n, random_state=random.randint(0, 9999))

    
    print(f"  RANDOM SAMPLE PREDICTIONS (n={n})")
   

    correct = 0
    for _, row in samples.iterrows():
        text   = str(row["combined_text"])
        source = str(row["source_domain"])
        true_label = int(row["label"])

        pred_label, confidence, probs = predict_single(model, text, source)

        true_str = "FAKE" if true_label == 1 else "REAL"
        pred_str = "FAKE" if pred_label == 1 else "REAL"
        match    = "✅" if pred_label == true_label else "❌"

        print(f"\n  {match} True: {true_str:<5} | Predicted: {pred_str:<5} | "
              f"Confidence: {confidence:.1f}%")
        print(f"     Text   : {text[:100]}...")
        print(f"     Source : {source}")

        if pred_label == true_label:
            correct += 1

    print(f"\n  Accuracy on {n} samples: {correct}/{n} = {correct/n*100:.1f}%")
    print("="*60)


def interactive_mode(model):
  
    print("  INTERACTIVE FAKE NEWS DETECTOR")
    print("  Type 'quit' to exit | 'random' for random samples")
    

    while True:
        print("\nOptions:")
        print("  [1] Enter your own news text")
        print("  [2] Test random samples from dataset")
        print("  [3] Quit")

        choice = input("\nChoose (1/2/3): ").strip()

        if choice == "1":
            text = input("\nEnter news headline or article text:\n> ").strip()
            if not text:
                print("No text entered. Try again.")
                continue

            source = input("Enter source domain (press Enter to skip): ").strip()
            if not source:
                source = "unknown"

            pred_label, confidence, probs = predict_single(model, text, source)
            print_result(text, source, pred_label, confidence, probs)

        elif choice == "2":
            try:
                n = input("How many random samples? (default=5): ").strip()
                n = int(n) if n.isdigit() else 5
                n = max(1, min(n, 50))   # clamp between 1 and 50
            except Exception:
                n = 5
            test_random_samples(model, n=n)

        elif choice == "3":
            print("\nExiting. Goodbye!")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    model = load_model(model_path="best")
    interactive_mode(model)