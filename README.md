# Detect Malicious Users Using Metric Learning — Educational Reconstruction

This repository contains a **simplified, newly written demonstration**, inspired by the academic presentation *Detect Professional Malicious User with Metric Learning in Recommender Systems*. **It is not the original project source code and does not reproduce the paper's MMD model or reported results.**

## What it demonstrates

- Rating and review ingestion (`user_id`, `rating`, `review`)
- Simple **lexicon-based** review sentiment (not an RNN)
- User-level behavior profiles and rating/review disagreement
- Unsupervised covariance-based Mahalanobis metric transformation (whitening)
- DBSCAN clustering and heuristic suspicious-user flags

## Run

```bash
pip install -r requirements.txt
python train.py
```

This generates `results.csv` using built-in **synthetic example data**. To use your own CSV:

```bash
python train.py --input your_reviews.csv --output results.csv
```

Required CSV columns: `user_id`, `rating` (1–5), `review`.

## Limitations

The lexicon sentiment approach is deliberately simple and may fail on sarcasm, negation, and complex language. Flags are **not ground-truth labels or proof of malicious behavior**. This example has no benchmark evaluation, no attention mechanism, and no modified RNN. For a faithful reproduction, obtain the original dataset, code, training details, and evaluation protocol.
