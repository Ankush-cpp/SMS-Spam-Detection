"""
text_utils.py
=============
Shared text-preprocessing utilities used by both train_model.py and app.py.
Keeping this in a separate module ensures the TF-IDF pipeline's
``preprocessor`` function can be properly pickled and unpickled regardless
of which script is the entry point.
"""

import re


def clean_text(text: str) -> str:
    """Lowercase, normalise URLs/numbers, strip punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # URLs → token
    text = re.sub(r"\d+", " num ", text)               # digits → token
    text = re.sub(r"[^a-z\s]", " ", text)              # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text
