import re

SPAM_WORDS = {
    "win", "winner", "congratulations", "free", "deal", "gift", "click",
    "urgent", "limited", "offer", "now", "money", "bonus", "credit", "prize",
    "million", "crypto", "btc", "guarantee", "buy", "cheap"
}

def extract_features_from_email_text(text: str):
    """
    Compute simple features expected by the dataset:
    - words: total token count
    - links: http/https count
    - capital_words: ALL CAPS tokens (len >= 2)
    - spam_word_count: tokens matching SPAM_WORDS
    """
    tokens = re.findall(r"[A-Za-z0-9@._:-]+", text)
    words = len(tokens)

    links = len(re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE))

    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

    lowered = [t.lower() for t in tokens]
    spam_word_count = sum(1 for t in lowered if t in SPAM_WORDS)

    return {
        "words": float(words),
        "links": float(links),
        "capital_words": float(capital_words),
        "spam_word_count": float(spam_word_count),
    }

def load_feature_schema(df_columns):
    # kept for compatibility; main alignment happens in app.py
    return [c for c in ["words", "links", "capital_words", "spam_word_count"] if c in df_columns]
