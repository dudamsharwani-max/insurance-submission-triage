"""
Business description -> NAICS + class code.

TF-IDF over word and character n-grams into a linear classifier. Character
n-grams are what keep it standing up to broker shorthand ("comml janitorial
svcs") and transposed-letter typos, which word features alone miss.

Evaluation split is by PHRASING, not by row: one description template per
class is held out of training entirely, so unseen-wording accuracy is
reported separately from the easier seen-wording case. Splitting by row
would leak templates across the split and inflate the score.
"""

import json
import pickle
import zlib
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "class_model.pkl"

CONFIDENCE_FLOOR = 0.55  # below this, the classifier abstains -> referral


def build_pipeline():
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, min_df=2)),
        ])),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced"),
            cv=3, method="sigmoid")),
    ])


def load_rows():
    with open(ROOT / "data" / "submissions.jsonl") as f:
        return [json.loads(l) for l in f]


def phrasing_split(rows):
    """Hold out one description template per class for the unseen-wording test."""
    templates = {}
    for r in rows:
        templates.setdefault(r["naics"], set()).add(r["description"])
    heldout = {naics: sorted(t)[-1] for naics, t in templates.items()}
    train, test_seen, test_unseen = [], [], []
    for r in rows:
        if r["description"] == heldout[r["naics"]]:
            test_unseen.append(r)
        elif zlib.crc32(r["submission_id"].encode()) % 5 == 0:
            test_seen.append(r)
        else:
            train.append(r)
    return train, test_seen, test_unseen


def train(verbose=True):
    rows = load_rows()
    tr, te_seen, te_unseen = phrasing_split(rows)

    Xtr = [r["description_as_written"] for r in tr]
    ytr = [r["naics"] for r in tr]
    pipe = build_pipeline().fit(Xtr, ytr)

    report = {"n_train": len(tr)}
    for name, split in [("seen_wording", te_seen), ("unseen_wording", te_unseen)]:
        if not split:
            continue
        X = [r["description_as_written"] for r in split]
        y = [r["naics"] for r in split]
        pred = pipe.predict(X)
        acc = float(np.mean(np.array(pred) == np.array(y)))
        report[name] = {"n": len(split), "accuracy": round(acc, 4)}
        if verbose:
            print(f"\n=== {name} (n={len(split)}) accuracy={acc:.3f} ===")
            print(classification_report(y, pred, zero_division=0, digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    with open(ROOT / "outputs" / "classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return pipe, report


_cache = None


def get_model():
    global _cache
    if _cache is None:
        with open(MODEL_PATH, "rb") as f:
            _cache = pickle.load(f)
    return _cache


# NAICS -> ISO-style class code lookup, derived from the taxonomy.
def naics_to_class_code():
    from generate_submissions import CLASSES
    return {naics: code for _, naics, code, *_ in CLASSES}


_CODES = None


def predict(description):
    """Return (naics, class_code, confidence). naics is None if below floor."""
    global _CODES
    if _CODES is None:
        _CODES = naics_to_class_code()
    if not description:
        return None, None, 0.0
    pipe = get_model()
    proba = pipe.predict_proba([description])[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    naics = pipe.classes_[idx]
    if conf < CONFIDENCE_FLOOR:
        return None, None, conf
    return naics, _CODES.get(naics), conf


if __name__ == "__main__":
    train()
