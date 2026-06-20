import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

FAKENEWSNET_PATH = os.path.join(DATA_DIR, "FakeNewsNet.csv")
WELFAKE_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
POLITIFACT_PATH = os.path.join(DATA_DIR, "politifact_factcheck_data.json")



TRANSFORMER_MODEL = "roberta-base"   # or "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
DROPOUT = 0.3
HIDDEN_DIM = 256
NUM_CLASSES = 2
SEED = 42


CREDIBILITY_DIM = 4
SENTIMENT_DIM = 6
TRANSFORMER_DIM = 768

FUSED_DIM = TRANSFORMER_DIM + CREDIBILITY_DIM + SENTIMENT_DIM



TRUSTED_SOURCES = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "nytimes.com": 0.85,
    "theguardian.com": 0.85,
    "washingtonpost.com": 0.85,
    "npr.org": 0.88,
    "politifact.com": 0.90,
    "snopes.com": 0.88,
    "usatoday.com": 0.80,
    "people.com": 0.70,
    "today.com": 0.72,
    "etonline.com": 0.55,
    "toofab.com": 0.40,
    "dailymail.co.uk": 0.45,
    "zerchoo.com": 0.30,
    "popsugar.com": 0.60,
    "telegraph.co.uk": 0.75,
}