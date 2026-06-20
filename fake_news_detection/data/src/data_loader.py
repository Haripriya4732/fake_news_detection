import pandas as pd
import json
import os
from src.config import FAKENEWSNET_PATH, WELFAKE_PATH, POLITIFACT_PATH


def load_welfake():

    df = pd.read_csv(WELFAKE_PATH)
    df = df[["title", "text", "label"]].copy()
    df.dropna(subset=["text"], inplace=True)
    df["label"] = df["label"].astype(int)
    df["source_domain"] = "unknown"
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["dataset"] = "welfake"
    return df[["combined_text", "source_domain", "label", "dataset"]]


def load_fakenewsnet():

    df = pd.read_csv(FAKENEWSNET_PATH)
    df = df[["title", "source_domain", "real"]].copy()
    df.dropna(subset=["title"], inplace=True)
    df["label"] = (1 - df["real"]).astype(int)   # invert: fake=1
    df["source_domain"] = df["source_domain"].fillna("unknown")
    df["combined_text"] = df["title"].fillna("")
    df["dataset"] = "fakenewsnet"
    return df[["combined_text", "source_domain", "label", "dataset"]]


def load_politifact():

    try:
        with open(POLITIFACT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception:
        df = pd.read_json(POLITIFACT_PATH, lines=True)

    FAKE_VERDICTS = {"false", "pants-fire", "mostly-false", "barely-true"}

    df["label"] = df["verdict"].str.lower().apply(
        lambda v: 1 if v in FAKE_VERDICTS else 0
    )
    df["combined_text"] = df["statement"].fillna("")
    df["source_domain"] = df["statement_source"].fillna("unknown")
    df["dataset"] = "politifact"
    return df[["combined_text", "source_domain", "label", "dataset"]]


def load_all_datasets():
  
    print("Loading WELFake...")
    welfake = load_welfake()

    print("Loading FakeNewsNet...")
    fakenewsnet = load_fakenewsnet()

    print("Loading PolitiFact...")
    politifact = load_politifact()

    combined = pd.concat([welfake, fakenewsnet, politifact], ignore_index=True)
    combined.dropna(subset=["combined_text", "label"], inplace=True)
    combined["combined_text"] = combined["combined_text"].astype(str).str.strip()
    combined = combined[combined["combined_text"].str.len() > 10]

    print(f"\nCombined Dataset Shape: {combined.shape}")
    print(f"Label Distribution:\n{combined['label'].value_counts()}")

    return combined