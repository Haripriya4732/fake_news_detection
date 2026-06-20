import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from src.config import PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "that", "this", "was", "are", "be", "been", "with",
    "have", "has", "had", "not", "but", "as", "by", "from", "he", "she",
    "they", "we", "you", "i", "his", "her", "their", "our", "its",
    "will", "would", "could", "should", "said", "says", "say",
    "news", "according", "also", "more", "after", "about", "than",
    "which", "who", "what", "when", "where", "how", "were", "so"
}


def tokenize(text: str):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]


def get_top_words(texts, n=40):
    all_words = []
    for t in texts:
        all_words.extend(tokenize(t))
    return Counter(all_words).most_common(n)


def plot_word_frequency(df):
    """Bar plots of top words in Fake vs Real news."""
    fake_texts = df[df["label"] == 1]["combined_text"].tolist()
    real_texts = df[df["label"] == 0]["combined_text"].tolist()

    fake_words = get_top_words(fake_texts, 30)
    real_words = get_top_words(real_texts, 30)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Fake
    words_f, counts_f = zip(*fake_words)
    axes[0].barh(list(reversed(words_f)), list(reversed(counts_f)), color="tomato")
    axes[0].set_title("Top Words in FAKE News", fontsize=14)
    axes[0].set_xlabel("Frequency")

    # Real
    words_r, counts_r = zip(*real_words)
    axes[1].barh(list(reversed(words_r)), list(reversed(counts_r)), color="steelblue")
    axes[1].set_title("Top Words in REAL News", fontsize=14)
    axes[1].set_xlabel("Frequency")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "word_frequency_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Word frequency comparison saved: {save_path}")


def plot_wordclouds(df):
    """Word clouds for Fake and Real news."""
    fake_texts = " ".join(df[df["label"] == 1]["combined_text"].tolist())
    real_texts = " ".join(df[df["label"] == 0]["combined_text"].tolist())

    fake_tokens = " ".join(tokenize(fake_texts))
    real_tokens = " ".join(tokenize(real_texts))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    wc_fake = WordCloud(
        width=700, height=400,
        background_color="white",
        colormap="Reds",
        max_words=100
    ).generate(fake_tokens)

    wc_real = WordCloud(
        width=700, height=400,
        background_color="white",
        colormap="Blues",
        max_words=100
    ).generate(real_tokens)

    axes[0].imshow(wc_fake, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("FAKE News Word Cloud", fontsize=14)

    axes[1].imshow(wc_real, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("REAL News Word Cloud", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "wordclouds.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Word clouds saved: {save_path}")


def plot_exclusive_words(df):
    """Words that appear more in Fake vs Real (ratio-based)."""
    fake_texts = df[df["label"] == 1]["combined_text"].tolist()
    real_texts = df[df["label"] == 0]["combined_text"].tolist()

    fake_counter = Counter()
    real_counter = Counter()

    for t in fake_texts:
        fake_counter.update(tokenize(t))
    for t in real_texts:
        real_counter.update(tokenize(t))

    all_words = set(fake_counter.keys()) | set(real_counter.keys())

    ratios = []
    for word in all_words:
        fc = fake_counter.get(word, 0) + 1
        rc = real_counter.get(word, 0) + 1
        total = fc + rc
        if total < 20:
            continue
        ratios.append((word, fc / rc, total))

    ratios.sort(key=lambda x: x[1], reverse=True)

    top_fake_biased = ratios[:20]
    top_real_biased = sorted(ratios, key=lambda x: x[1])[:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    words_f, scores_f, _ = zip(*top_fake_biased)
    axes[0].barh(list(reversed(words_f)), list(reversed(scores_f)), color="tomato")
    axes[0].set_title("Words More Common in FAKE News\n(Fake/Real ratio)", fontsize=12)
    axes[0].set_xlabel("Fake/Real Ratio")

    words_r, scores_r, _ = zip(*top_real_biased)
    real_ratios_inv = [1/s for s in scores_r]
    axes[1].barh(list(reversed(words_r)), list(reversed(real_ratios_inv)), color="steelblue")
    axes[1].set_title("Words More Common in REAL News\n(Real/Fake ratio)", fontsize=12)
    axes[1].set_xlabel("Real/Fake Ratio")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "exclusive_word_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Exclusive word analysis saved: {save_path}")


def run_word_analysis(df):
    print("\nRunning Word Analysis...")
    plot_word_frequency(df)
    plot_wordclouds(df)
    plot_exclusive_words(df)
    print("Word analysis complete.")