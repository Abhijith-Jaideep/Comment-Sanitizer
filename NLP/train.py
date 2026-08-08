"""Rebuild the two pickles NLP.py loads at import time.

The trained artefacts were never committed, so the service could not start.
This script regenerates them from the Jigsaw Toxic Comment Classification
training set.

Two things about NLP.py constrain what can be trained here:

  result = sum(y_test_pred[0])
  if result >= 1: ...

Summing the probability vector and thresholding at 1 only makes sense for a
multi-label model, where each of the six labels carries an independent
probability and the sum runs 0..6. A plain binary classifier returns two
columns that sum to exactly 1.0, so that test would flag every comment as
Negative. So the model has to stay a OneVsRest over the six Jigsaw labels,
and the vectorizer has to be a separate object, because NLP.py pickles them
apart and calls td.transform() then mp.predict_proba().

clean_text is duplicated from NLP.py rather than imported: importing it would
execute NLP.py's module level pickle loads, which is exactly what does not
exist yet on a fresh checkout.

Usage:
    python train.py --data train.csv
"""

import argparse
import pickle
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# The gate NLP.py applies to the summed probability vector.
DECISION_THRESHOLD = 1.0


def clean_text(text):
    """Character for character the same as NLP.py's clean_text.

    Train time and predict time preprocessing have to agree exactly, or the
    vocabulary learned here will not line up with what the service feeds in.
    """
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub("(\\W)", " ", text)
    text = re.sub("\\S*\\d\\S*\\s*", "", text)
    return text


def load(path):
    df = pd.read_csv(path)

    missing = [c for c in ["comment_text"] + LABELS if c not in df.columns]
    if missing:
        sys.exit(
            f"{path} is missing columns: {', '.join(missing)}\n"
            "Expected the Jigsaw Toxic Comment Classification train.csv, which "
            "has comment_text plus the six label columns."
        )

    df = df.dropna(subset=["comment_text"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.csv")
    ap.add_argument("--out-model", default="model_pickle")
    ap.add_argument("--out-vectorizer", default="vectorizer.pk")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"No dataset at {data_path.resolve()}")

    print(f"Reading {data_path} ...")
    df = load(data_path)
    print(f"  {len(df):,} rows")
    for label in LABELS:
        n = int(df[label].sum())
        print(f"  {label:<14} {n:>7,}  ({n / len(df):.2%})")

    print("\nCleaning text ...")
    t0 = time.time()
    X = df.comment_text.apply(clean_text)
    y = df[LABELS].values
    print(f"  done in {time.time() - t0:.1f}s")

    # Stratify on "is this comment toxic at all", so the rarer labels do not
    # end up lopsided between the splits.
    any_toxic = (y.sum(axis=1) > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=any_toxic
    )
    print(f"\nTrain {len(X_train):,} / test {len(X_test):,}")

    print("\nFitting vectorizer ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # liblinear sorts a matrix's indices in place before fitting. Leave the
    # matrices non canonical and that write lands on a read only buffer,
    # failing with "WRITEBACKIFCOPY base is read-only". Doing it here once,
    # on a writable array, keeps the fit read only as it expects.
    X_train_vec.sum_duplicates()
    X_test_vec.sum_duplicates()
    print(f"  {len(vectorizer.vocabulary_):,} features in {time.time() - t0:.1f}s")

    print("\nTraining ...")
    t0 = time.time()
    # Deliberately single process: six liblinear fits on sparse text take
    # seconds each, and n_jobs > 1 makes joblib memmap the matrix read only
    # into workers, which is what triggered the failure above.
    model = OneVsRestClassifier(
        LogisticRegression(C=4.0, solver="liblinear", max_iter=1000),
    )
    model.fit(X_train_vec, y_train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nPer label ROC AUC:")
    probs = model.predict_proba(X_test_vec)
    aucs = []
    for i, label in enumerate(LABELS):
        # AUC is undefined for a label with no positives in the split. The
        # rare labels (threat is well under 1% of Jigsaw) can hit this on a
        # small or unlucky sample, and it should not take the run down.
        if len(np.unique(y_test[:, i])) < 2:
            print(f"  {label:<14}      n/a  (no positives in test split)")
            continue
        auc = roc_auc_score(y_test[:, i], probs[:, i])
        aucs.append(auc)
        print(f"  {label:<14} {auc:.4f}")
    if aucs:
        print(f"  {'mean':<14} {np.mean(aucs):.4f}  (over {len(aucs)} labels)")

    # What the service will actually do: sum the six probabilities, flag at 1.
    print(f"\nService gate (sum of probabilities >= {DECISION_THRESHOLD}):")
    pred_flagged = (probs.sum(axis=1) >= DECISION_THRESHOLD).astype(int)
    true_flagged = (y_test.sum(axis=1) > 0).astype(int)
    print(
        classification_report(
            true_flagged,
            pred_flagged,
            target_names=["clean", "toxic"],
            digits=4,
            zero_division=0,
        )
    )

    # The threshold of 1 in NLP.py is arbitrary rather than fitted, so it is
    # worth seeing what it costs. Printed as a diagnostic only: changing it
    # means editing NLP.py, which this script does not touch.
    print("Where the threshold sits:")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>7}  {'F1':>7}")
    totals = probs.sum(axis=1)
    best = (0.0, None)
    for t in [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]:
        flagged = (totals >= t).astype(int)
        tp = int(((flagged == 1) & (true_flagged == 1)).sum())
        fp = int(((flagged == 1) & (true_flagged == 0)).sum())
        fn = int(((flagged == 0) & (true_flagged == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, t)
        mark = "  <- NLP.py" if t == DECISION_THRESHOLD else ""
        print(f"  {t:>7.2f}  {prec:>9.4f}  {rec:>7.4f}  {f1:>7.4f}{mark}")
    print(f"\n  best F1 at threshold {best[1]} ({best[0]:.4f})\n")

    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)
    with open(args.out_vectorizer, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Wrote {args.out_model} and {args.out_vectorizer}")


if __name__ == "__main__":
    main()
