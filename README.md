# Fake News Detection Using Transformer-Based NLP with Source Credibility and Sentiment Fusion

## About
AI-powered fake news detector combining BERT/RoBERTa, source credibility scoring, and sentiment analysis for accurate misinformation classification.

---

## Project Structure

```
fake_news_detection/
├── data/
│   ├── FakeNewsNet.csv
│   ├── WELFake_Dataset.csv
│   └── politifact_factcheck_data.json
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── source_credibility.py
│   ├── sentiment_module.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── word_analysis.py
├── main.py
├── test_interactive.py
├── run.bat
└── requirements.txt
```

---

## How to Run (Step by Step)

### Step 1 — Open Anaconda Prompt
Search **Anaconda Prompt** in the Windows Start menu and open it.

### Step 2 — Activate Your Environment
```bash
conda activate your_env_name
```
> Skip this if you are using the base environment.

### Step 3 — Go to Project Folder
```bash
cd C:\Users\krake\Downloads\fake_news\fake_news_detection
```

### Step 4 — Install All Dependencies
```bash
pip install -r requirements.txt
```
> Only needed once. This installs torch, transformers, shap, lime, etc.

### Step 5 — Train the Model
```bash
python main.py
```
> This will load all 3 datasets, build features, train the model, save it,
> evaluate it, and generate all plots inside the `outputs/` folder.

### Step 6 — Test Interactively
```bash
python test_interactive.py
```
> After this runs, you will see a menu:
> - Option 1 → Type your own news text and get a prediction
> - Option 2 → Pick random samples from the dataset and test
> - Option 3 → Quit

---

## Datasets

| Dataset | Rows | Label |
|---|---|---|
| WELFake | 72,134 | 1=Fake, 0=Real |
| FakeNewsNet | 23,196 | 1=Fake, 0=Real |
| PolitiFact | 21,152 | Verdict-based |

---

## Features

- RoBERTa / BERT transformer backbone
- Source credibility trust scoring
- VADER + TextBlob sentiment analysis
- Adaptive feature weighting and fusion
- SHAP and LIME explainability
- Word cloud and frequency analysis
- Model saving and interactive testing

---

## Outputs Generated

| File | Description |
|---|---|
| `outputs/models/best_model.pt` | Best saved model |
| `outputs/plots/training_curves.png` | Loss and accuracy graphs |
| `outputs/plots/confusion_matrix.png` | Confusion matrix |
| `outputs/plots/roc_curve.png` | ROC curve |
| `outputs/plots/wordclouds.png` | Fake vs Real word clouds |
| `outputs/plots/lime_sample_*.png` | LIME explanations |
| `outputs/plots/shap_summary.png` | SHAP summary |
| `outputs/results/classification_report.txt` | Precision, Recall, F1 |
