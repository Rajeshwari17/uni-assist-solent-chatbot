"""Uni-Assist v2: an intent-classification chatbot for university website queries.

Design choices
- TF-IDF on character n-grams + word n-grams -> tolerant of typos.
- Logistic regression -> gives class probabilities used as a confidence score.
- Confidence threshold -> below it the bot says it is unsure and offers a human contact.
- Sensitive intents (wellbeing, harassment) always return a fixed, safe response
  and are matched at a LOWER threshold, because missing them is worse than a false alarm.
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).parent
THRESHOLD = 0.35            # general confidence threshold (tuned by evaluate.py)
SENSITIVE_THRESHOLD = 0.20  # lower on purpose for safety-critical intents

FALLBACK = ("I'm not sure I understood that, so I don't want to guess. "
            "Please try rephrasing, or contact the Student Hub: "
            "student.hub@solent.ac.uk, 023 8201 5200.")


def normalise(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9' ]+", " ", text)


def load_intents(path=HERE / "intents.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intents"]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=20, max_iter=2000)
    return Pipeline([("features", features), ("clf", clf)])


def training_data(intents):
    X, y = [], []
    for it in intents:
        for p in it["patterns"]:
            X.append(normalise(p))
            y.append(it["tag"])
    return X, y


def train(intents=None):
    intents = intents or load_intents()
    X, y = training_data(intents)
    model = build_pipeline()
    model.fit(X, y)
    return model


class UniAssist:
    def __init__(self, threshold=THRESHOLD, sensitive_threshold=SENSITIVE_THRESHOLD):
        self.intents = load_intents()
        self.by_tag = {it["tag"]: it for it in self.intents}
        self.model = train(self.intents)
        self.threshold = threshold
        self.sensitive_threshold = sensitive_threshold

    def predict(self, text: str):
        """Return (tag_or_None, confidence). None means 'not sure'."""
        probs = self.model.predict_proba([normalise(text)])[0]
        classes = self.model.classes_
        order = np.argsort(probs)[::-1]
        top = classes[order[0]]
        conf = float(probs[order[0]])

        # Safety first: if any sensitive intent clears its lower threshold, route there.
        for idx in order:
            tag = classes[idx]
            if self.by_tag[tag].get("sensitive") and probs[idx] >= self.sensitive_threshold:
                return tag, float(probs[idx])

        if conf >= self.threshold:
            return top, conf
        return None, conf

    def respond(self, text: str) -> str:
        tag, _ = self.predict(text)
        if tag is None:
            return FALLBACK
        return random.choice(self.by_tag[tag]["responses"])


if __name__ == "__main__":
    bot = UniAssist()
    print("Uni-Assist (type 'quit' to exit). I'm an automated assistant, not a person.")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"quit", "exit"}:
            break
        tag, conf = bot.predict(q)
        print(f"Bot: {bot.respond(q)}   [intent={tag}, confidence={conf:.2f}]")
