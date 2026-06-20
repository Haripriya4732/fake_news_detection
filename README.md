# Fake News Detection Using Transformer-Based NLP with Source Credibility and Sentiment Fusion

## Overview

This project presents an advanced **Fake News Detection System** that combines **Transformer-based Natural Language Processing (NLP)** with **Source Credibility Analysis** and **Sentiment Fusion** to accurately classify news articles as **Fake** or **Real**.

Traditional fake news detection approaches rely only on textual content. This framework enhances prediction reliability by integrating:

* Transformer-based contextual text understanding (BERT/RoBERTa)
* Source credibility scoring
* Sentiment analysis
* Explainable AI techniques (SHAP & LIME)

The system provides interpretable predictions and supports real-world misinformation detection applications.

---

# Key Features

Transformer-Based News Classification (BERT / RoBERTa)

Source Credibility Scoring Module

Sentiment Analysis using VADER and TextBlob

Feature Fusion Framework

Explainable AI (SHAP + LIME)

Word Cloud and Frequency Analysis

Model Performance Visualization

Interactive News Testing Interface

Automatic Model Saving and Loading

---

# Project Architecture

```text
News Article
      │
      ▼
Text Preprocessing
      │
      ▼
Transformer Encoder
(BERT / RoBERTa)
      │
      ├───────────────┐
      ▼               ▼
Sentiment       Source Credibility
Analysis           Scoring
      │               │
      └───────┬───────┘
              ▼
      Feature Fusion Layer
              ▼
      Fake / Real Prediction
              ▼
      Explainability Module
       (SHAP + LIME)
```

---

# Project Structure

```text
fake_news_detection/
│
├── data/
│   ├── FakeNewsNet.csv
│   ├── WELFake_Dataset.csv
│   └── politifact_factcheck_data.json
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
│
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
│
├── main.py
├── test_interactive.py
├── run.bat
├── requirements.txt
└── README.md
```

---

# Datasets Used

| Dataset            | Description                                      |
| ------------------ | ------------------------------------------------ |
| WELFake Dataset    | Large-scale fake and real news dataset           |
| FakeNewsNet        | News articles with social engagement information |
| PolitiFact Dataset | Fact-checking dataset with verdict labels        |

### Dataset Statistics

| Dataset     | Records |
| ----------- | ------- |
| WELFake     | 72,134  |
| FakeNewsNet | 23,196  |
| PolitiFact  | 21,152  |

Total Combined Samples: **116,482+**

---

# Technologies Used

### Programming Language

* Python 3.10+

### Deep Learning Frameworks

* PyTorch
* Hugging Face Transformers

### NLP Libraries

* BERT
* RoBERTa
* NLTK
* TextBlob
* VADER Sentiment

### Explainable AI

* SHAP
* LIME

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Matplotlib
* Seaborn
* WordCloud

---

# Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/Haripriya4732/fake_news_detection
cd fake-news-detection
```

## Step 2: Create Environment

```bash
conda create -n fake_news python=3.10

conda activate fake_news
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train the Model

```bash
python main.py
```

This process will:

* Load all datasets
* Perform preprocessing
* Generate sentiment features
* Calculate source credibility scores
* Train the transformer model
* Evaluate performance
* Save outputs automatically

---

## Interactive Testing

```bash
python test_interactive.py
```

Options:

```text
1. Enter custom news text
2. Test random samples
3. Exit
```

---

# Methodology

### 1. Data Collection

* WELFake Dataset
* FakeNewsNet Dataset
* PolitiFact Dataset

### 2. Text Preprocessing

* Lowercasing
* URL removal
* Special character cleaning
* Tokenization
* Stopword removal

### 3. Feature Extraction

#### Transformer Features

* BERT Embeddings
* RoBERTa Embeddings

#### Sentiment Features

* Positive Score
* Negative Score
* Neutral Score
* Compound Score

#### Source Features

* Domain Trust Score
* Historical Reliability
* Credibility Ranking

### 4. Feature Fusion

Combined Feature Vector:

```text
Fused Features =
Transformer Features
+
Sentiment Features
+
Credibility Features
```

### 5. Classification

Binary Classification:

```text
0 → Real News
1 → Fake News
```

---

# Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

# Generated Outputs

## Saved Model

```text
outputs/models/best_model.pt
```

---

## Visualization Outputs

```text
outputs/plots/training_curves.png
outputs/plots/confusion_matrix.png
outputs/plots/roc_curve.png
outputs/plots/wordclouds.png
outputs/plots/shap_summary.png
outputs/plots/lime_sample_*.png
```

---

## Evaluation Results

```text
outputs/results/classification_report.txt
```

Contains:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# Explainable AI

The framework incorporates:

### SHAP

Provides:

* Global feature importance
* Local prediction explanations

### LIME

Provides:

* Instance-level explanations
* Highlighted influential words

This improves model transparency and trustworthiness.

---

# Sample Prediction

### Input

```text
Scientists discover miracle cure that guarantees immortality.
```

### Output

```text
Prediction : Fake News

Confidence : 97.6%

Sentiment : Highly Positive

Source Credibility : Low
```

---

# Applications

* News Verification Platforms
* Social Media Monitoring
* Fact Checking Systems
* Journalism Support Tools
* Misinformation Detection
* Content Moderation

---

# Future Enhancements

* Multilingual Fake News Detection
* Real-Time Web Scraping
* Social Media Stream Analysis
* Knowledge Graph Integration
* Graph Neural Networks
* Federated Learning Deployment
* Web Dashboard Deployment

---


# License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for academic and research purposes.
