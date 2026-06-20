import os
import torch
import numpy as np
from src.config import SEED
from src.data_loader import load_all_datasets
from src.source_credibility import build_source_features
from src.sentiment_module import build_sentiment_features
from src.train import train_model
from src.evaluate import load_and_test
from src.explainability import lime_explain, shap_explain
from src.word_analysis import run_word_analysis

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
  
    print("FAKE NEWS DETECTION - MULTI-MODAL TRANSFORMER FRAMEWORK")
  


    print("\n[1] Loading Datasets...")
    df = load_all_datasets()


    print("\n[2] Word Analysis...")
    run_word_analysis(df)

    
    print("\n[3] Building Source Credibility Features...")
    source_features = build_source_features(df["source_domain"].tolist())

    print("[3] Building Sentiment Features...")
    sentiment_features = build_sentiment_features(df["combined_text"].tolist())

   
    print("\n[4] Training Model...")
    model, test_loader, X_test, y_test, src_test, sent_test = train_model(
        df, source_features, sentiment_features
    )

   
    print("\n[5] Evaluating Best Model...")
    all_preds, all_labels, all_probs = load_and_test(
        X_test, src_test, sent_test, y_test, model_path="best"
    )

   
    print("\n[6] LIME Explanations...")
    lime_explain(model, X_test[:10], df["source_domain"].tolist()[:10], y_test[:10], n_samples=5)

    print("\n[7] SHAP Explanations...")
    shap_explain(model, X_test[:10], df["source_domain"].tolist()[:10], y_test[:10], n_samples=5)

    
    print("PIPELINE COMPLETE./")
   


if __name__ == "__main__":
    main()