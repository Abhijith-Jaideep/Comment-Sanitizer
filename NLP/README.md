# NLP service

Flask service on port 7000. The backend posts a comment to it and gets back
`Positive` or `Negative`.

## Why train.py exists

`NLP.py` loads two pickles at import time:

```python
with open('model_pickle','rb') as file: mp = pickle.load(file)
with open('vectorizer.pk','rb') as file: td = pickle.load(file)
```

Neither was ever committed, so a fresh checkout cannot start the service at
all. `train.py` regenerates them.

## The model has to stay multi-label

`NLP.py` decides with:

```python
result = sum(y_test_pred[0])
if result >= 1: ...
```

That only works across independent per-label probabilities, where the sum runs
0 to 6. A plain binary classifier returns two columns summing to exactly 1.0,
so the same test would flag every comment as Negative. So the model stays a
`OneVsRestClassifier` over the six Jigsaw labels, and the vectorizer stays a
separate object because `NLP.py` unpickles them apart.

`clean_text` is duplicated in `train.py` rather than imported, because
importing from `NLP.py` would run its module level pickle loads, which is the
very thing that does not exist yet before training.

## Retraining

Get `train.csv` from the Jigsaw Toxic Comment Classification Challenge on
Kaggle and put it in this folder. It needs `comment_text` plus the six label
columns.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python train.py --data train.csv
.venv/Scripts/python verify.py
```

`train.py` writes `model_pickle` and `vectorizer.pk` next to `NLP.py`, and
prints per-label ROC AUC plus a report on the decision gate the service
actually uses. `verify.py` reloads the pickles and runs known clean and known
toxic comments through the same path the service takes, so a bad artefact
surfaces there instead of as a 500 at request time.

## Versions

`requirements.txt` is pinned. The artefacts are scikit-learn objects, and
unpickling across a different major version is not guaranteed to work. When it
breaks it breaks at import, taking the service down on startup rather than
failing one request. Train and serve on the same versions.
