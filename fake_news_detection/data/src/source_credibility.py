import numpy as np
from src.config import TRUSTED_SOURCES


def get_source_credibility_features(source_domain: str) -> np.ndarray:
    """
    Returns a 4-dim credibility feature vector:
    [trust_score, is_known, is_high_trust, is_low_trust]
    """
    domain = str(source_domain).lower().strip()

    # Match by suffix
    trust_score = 0.5  # default unknown
    for known_domain, score in TRUSTED_SOURCES.items():
        if known_domain in domain:
            trust_score = score
            break

    is_known = 1.0 if any(k in domain for k in TRUSTED_SOURCES) else 0.0
    is_high_trust = 1.0 if trust_score >= 0.80 else 0.0
    is_low_trust = 1.0 if trust_score <= 0.45 else 0.0

    return np.array([trust_score, is_known, is_high_trust, is_low_trust], dtype=np.float32)


def build_source_features(source_domains):
    return np.stack([get_source_credibility_features(d) for d in source_domains])