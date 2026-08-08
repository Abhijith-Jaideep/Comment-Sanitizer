"""Check the regenerated pickles satisfy the contract NLP.py depends on.

Runs the same path the service does (unpickle, clean, transform,
predict_proba, sum, threshold at 1) without starting Flask, so a broken
artefact shows up here rather than as a 500 at request time.
"""

import pickle
import sys

import pandas as pd

from train import DECISION_THRESHOLD, clean_text

CLEAN = [
    "Thanks for the detailed write up, this really helped me out.",
    "I disagree with your conclusion but the data is interesting.",
    "Could you explain the second paragraph a bit more?",
    "Great post, subscribed.",
    "This does not work on my machine, any ideas what I am missing?",
]

TOXIC = [
    "you are a complete idiot and everyone hates you",
    "shut up you moron nobody cares what you think",
    "i will find you and hurt you",
    "what a stupid worthless piece of garbage post",
    "get lost you pathetic loser",
]


def main():
    try:
        with open("model_pickle", "rb") as f:
            mp = pickle.load(f)
        with open("vectorizer.pk", "rb") as f:
            td = pickle.load(f)
    except FileNotFoundError as e:
        sys.exit(f"{e.filename} not found. Run train.py first.")

    print(f"model      {type(mp).__name__}")
    print(f"vectorizer {type(td).__name__}, {len(td.vocabulary_):,} features")

    n_out = len(getattr(mp, "estimators_", []))
    print(f"outputs    {n_out}")
    if n_out < 2:
        sys.exit(
            "FAIL: model has fewer than 2 outputs. NLP.py sums predict_proba "
            "and thresholds at 1, which needs independent per label "
            "probabilities. A binary model's rows sum to exactly 1.0 and would "
            "flag everything as Negative."
        )

    failures = 0
    for expected, comments in (("Positive", CLEAN), ("Negative", TOXIC)):
        print(f"\nExpect {expected}:")
        for c in comments:
            df = pd.DataFrame({"comment_text": [c]})
            df.comment_text = df.comment_text.apply(clean_text)
            probs = mp.predict_proba(td.transform(df.comment_text))
            total = sum(probs[0])
            got = "Negative" if total >= DECISION_THRESHOLD else "Positive"
            ok = "ok  " if got == expected else "FAIL"
            if got != expected:
                failures += 1
            print(f"  {ok} sum={total:5.2f}  {c[:58]}")

    print(f"\n{failures} unexpected of {len(CLEAN) + len(TOXIC)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
