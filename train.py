"""Educational reconstruction: unsupervised suspicious-reviewer screening.
Not the original MMD paper implementation; uses lexicon sentiment instead of an RNN.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

POSITIVE = {'great','good','excellent','love','amazing','perfect','happy','fast','useful','best'}
NEGATIVE = {'bad','poor','terrible','awful','worst','broken','slow','hate','disappointed','useless'}

def sentiment(review):
    words = set(''.join(c.lower() if c.isalpha() else ' ' for c in str(review)).split())
    return (len(words & POSITIVE) - len(words & NEGATIVE)) / max(1, len(words & (POSITIVE | NEGATIVE)))

def sample_data():
    rows = []
    reviews = ['great useful product', 'excellent good quality', 'bad broken product',
               'terrible slow service', 'good fast delivery', 'poor useless item']
    for uid in range(1, 31):
        for j in range(5):
            review = reviews[(uid + j) % len(reviews)]
            score = sentiment(review)
            rating = (5 if score > 0 else 1)
            if uid in (7, 14, 21) and j < 4:
                rating = 5 if score < 0 else 1  # deliberate rating-review mismatch
            rows.append({'user_id': f'user_{uid:02d}', 'rating': rating, 'review': review})
    return pd.DataFrame(rows)

def analyze(df, eps=1.8):
    required = {'user_id', 'rating', 'review'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV needs columns: {sorted(required)}')
    df = df.copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='raise')
    if df['rating'].isna().any() or not df['rating'].between(1, 5).all():
        raise ValueError('Ratings must be between 1 and 5, with no missing values.')
    if df[['user_id', 'review']].isna().any().any():
        raise ValueError('user_id and review cannot be missing.')
    df['sentiment'] = df['review'].map(sentiment)
    df['rating_norm'] = (df['rating'] - 3) / 2
    df['mismatch'] = (df['rating_norm'] - df['sentiment']).abs()
    features = df.groupby('user_id').agg(
        review_count=('review', 'size'),
        mean_rating=('rating', 'mean'),
        mean_sentiment=('sentiment', 'mean'),
        mean_mismatch=('mismatch', 'mean'),
        rating_variability=('rating', 'std'),
    ).fillna(0).reset_index()
    cols = ['review_count','mean_rating','mean_sentiment','mean_mismatch','rating_variability']
    x = StandardScaler().fit_transform(features[cols])
    # Learn an unsupervised Mahalanobis metric from covariance (whitening).
    # This is NOT attention-based metric learning or the original MMD algorithm.
    covariance = LedoitWolf().fit(x).covariance_
    eigvals, eigvecs = np.linalg.eigh(covariance)
    transformed = x @ eigvecs @ np.diag(1 / np.sqrt(np.maximum(eigvals, 1e-8)))
    clusters = DBSCAN(eps=eps, min_samples=3).fit_predict(transformed)
    features['cluster'] = clusters
    features['suspicious_flag'] = (clusters == -1) | (features['mean_mismatch'] >= 1.5)
    return features.sort_values(['suspicious_flag','mean_mismatch'], ascending=False)

def main():
    parser = argparse.ArgumentParser(description='Educational malicious-user detection demo')
    parser.add_argument('--input', help='Optional CSV with user_id,rating,review columns')
    parser.add_argument('--output', default='results.csv')
    parser.add_argument('--eps', type=float, default=1.8)
    args = parser.parse_args()
    if args.eps <= 0:
        parser.error('--eps must be positive')
    df = pd.read_csv(args.input) if args.input else sample_data()
    result = analyze(df, args.eps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result[['user_id','mean_mismatch','cluster','suspicious_flag']].to_string(index=False))
    print(f'\nSaved results to {out}')
    print('Flags are heuristic signals, not verified malicious-user labels.')

if __name__ == '__main__':
    main()
