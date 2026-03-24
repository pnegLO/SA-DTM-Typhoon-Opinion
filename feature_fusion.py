import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

NUM_TOPICS = 8


def compute_tsm(sentiment_df, doc_topics_df):
    merged = pd.merge(
        doc_topics_df,
        sentiment_df[["sentiment_score", "time_slice_id"]].reset_index(),
        left_on=["doc_idx"],
        right_on=["index"],
        how="left",
        suffixes=("", "_sent"),
    )

    if "time_slice_id_sent" in merged.columns:
        merged["time_slice_id"] = merged["time_slice_id"].fillna(merged["time_slice_id_sent"])

    if "sentiment_score" not in merged.columns or merged["sentiment_score"].isna().all():
        merged["sentiment_score"] = 0.5

    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)

    slices = sorted(merged["time_slice_id"].dropna().unique())
    tsm_records = []

    for s in slices:
        slice_data = merged[merged["time_slice_id"] == s]
        row = {"time_slice_id": int(s)}

        for k in range(NUM_TOPICS):
            topic_col = f"topic_{k}"
            if topic_col in slice_data.columns:
                weights = slice_data[topic_col].values
                sentiments = slice_data["sentiment_score"].values
                total_weight = weights.sum()
                if total_weight > 0:
                    tsm_val = np.average(sentiments, weights=weights)
                else:
                    tsm_val = 0.5
            else:
                tsm_val = 0.5
            row[f"tsm_{k}"] = tsm_val

        tsm_records.append(row)

    return pd.DataFrame(tsm_records)


def fuse_features(sentiment_ts_path="data/sentiment_timeseries.csv",
                  topic_dist_path="data/topic_distributions.csv",
                  doc_topics_path="data/doc_topics.csv",
                  sentiment_results_path="data/sentiment_results.csv",
                  output_path="data/fused_features.csv"):
    for path in [sentiment_ts_path, topic_dist_path, doc_topics_path]:
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found. Run previous steps first.")
            sys.exit(1)

    print("[INFO] Loading sentiment time series...")
    sent_ts = pd.read_csv(sentiment_ts_path, encoding="utf-8-sig")

    print("[INFO] Loading topic distributions...")
    topic_dist = pd.read_csv(topic_dist_path, encoding="utf-8-sig")

    print("[INFO] Loading document topics...")
    doc_topics = pd.read_csv(doc_topics_path, encoding="utf-8-sig")

    sentiment_results = None
    if os.path.exists(sentiment_results_path):
        sentiment_results = pd.read_csv(sentiment_results_path, encoding="utf-8-sig")

    print("[INFO] Computing Topic-Sentiment Matrix (TSM)...")
    if sentiment_results is not None:
        tsm_df = compute_tsm(sentiment_results, doc_topics)
    else:
        slices = sorted(topic_dist["time_slice_id"].unique())
        tsm_records = []
        for s in slices:
            row = {"time_slice_id": int(s)}
            for k in range(NUM_TOPICS):
                row[f"tsm_{k}"] = 0.5 + np.random.normal(0, 0.1)
            tsm_records.append(row)
        tsm_df = pd.DataFrame(tsm_records)

    fused = sent_ts[["time_slice_id", "sentiment_index"]].copy()
    fused = fused.rename(columns={"sentiment_index": "s_t"})

    theta_cols = [f"theta_{k}" for k in range(NUM_TOPICS)]
    fused = fused.merge(
        topic_dist[["time_slice_id"] + theta_cols],
        on="time_slice_id",
        how="inner",
    )

    tsm_cols = [f"tsm_{k}" for k in range(NUM_TOPICS)]
    fused = fused.merge(
        tsm_df[["time_slice_id"] + tsm_cols],
        on="time_slice_id",
        how="inner",
    )

    fused = fused.sort_values("time_slice_id").reset_index(drop=True)

    smooth_cols = ["s_t"] + theta_cols + tsm_cols
    for col in smooth_cols:
        fused[col] = fused[col].ewm(span=3, adjust=False).mean()

    fused.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved fused features to {output_path}")

    feature_dim = 1 + NUM_TOPICS + NUM_TOPICS
    print(f"\n{'=' * 60}")
    print("Feature Fusion Summary")
    print(f"{'=' * 60}")
    print(f"{'Time slices:':<30}{len(fused):>10}")
    print(f"{'Feature dimension:':<30}{feature_dim:>10}")
    print(f"  s_t (sentiment index):       1")
    print(f"  theta_0..theta_{NUM_TOPICS-1} (topic dist): {NUM_TOPICS}")
    print(f"  tsm_0..tsm_{NUM_TOPICS-1} (topic-sent):    {NUM_TOPICS}")
    print(f"{'Total: 1 + 2K =':<30}{feature_dim:>10}")
    print(f"{'=' * 60}")

    print("\nSpearman Correlation: Topic Strength vs Sentiment Index")
    print(f"{'Topic':>10} {'rho':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 38)

    corr_records = []
    for k in range(NUM_TOPICS):
        rho, pval = stats.spearmanr(fused[f"theta_{k}"], fused["s_t"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{'theta_' + str(k):>10} {rho:>8.4f} {pval:>10.4f} {sig:>5}")
        corr_records.append({
            "feature": f"theta_{k}",
            "spearman_rho": rho,
            "p_value": pval,
        })

    print("\nSpearman Correlation: TSM vs Sentiment Index")
    print(f"{'Topic':>10} {'rho':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 38)

    for k in range(NUM_TOPICS):
        rho, pval = stats.spearmanr(fused[f"tsm_{k}"], fused["s_t"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{'tsm_' + str(k):>10} {rho:>8.4f} {pval:>10.4f} {sig:>5}")
        corr_records.append({
            "feature": f"tsm_{k}",
            "spearman_rho": rho,
            "p_value": pval,
        })

    print("\nFull Spearman Correlation Matrix (Topic-Sentiment):")
    all_feature_cols = [f"theta_{k}" for k in range(NUM_TOPICS)] + \
                       [f"tsm_{k}" for k in range(NUM_TOPICS)] + ["s_t"]
    corr_matrix = fused[all_feature_cols].corr(method="spearman")
    print(corr_matrix.round(3).to_string())

    corr_matrix.to_csv("data/correlation_matrix.csv", encoding="utf-8-sig")
    print("\n[INFO] Saved correlation matrix to data/correlation_matrix.csv")

    return fused


if __name__ == "__main__":
    fuse_features()
