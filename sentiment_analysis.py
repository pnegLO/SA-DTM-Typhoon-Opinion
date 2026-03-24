import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)

try:
    from snownlp import SnowNLP
    USE_SNOWNLP = True
except ImportError:
    print("[WARN] SnowNLP not installed, using dictionary-based fallback.")
    USE_SNOWNLP = False


POSITIVE_WORDS = set(
    "安全 平安 感谢 致敬 加油 希望 好 支援 救援 保障 恢复 正常 "
    "有序 及时 迅速 高效 温暖 团结 坚强 勇敢 感动 爱心 捐 援助 "
    "到位 踏实 放心 顺利 成功 保护 稳定 改善 积极 有效 妥善 "
    "转移 安置 保重 辛苦 点赞 棒 厉害 英雄 逆行者".split()
)

NEGATIVE_WORDS = set(
    "害怕 恐惧 损失 受损 倒塌 中断 停电 停水 淹没 积水 被困 "
    "危险 紧急 严重 灾害 破坏 损毁 恶劣 暴雨 狂风 可怕 糟糕 "
    "担心 焦虑 恐慌 死亡 伤亡 失联 惨 崩溃 绝望 无助 悲伤 "
    "困难 艰难 受灾 险情 威胁 侵袭 肆虐 摧毁 冲毁".split()
)


def snownlp_sentiment(text):
    try:
        s = SnowNLP(text)
        return s.sentiments
    except Exception:
        return 0.5


def dict_sentiment(text, tokens_str=None):
    if tokens_str and isinstance(tokens_str, str):
        words = tokens_str.split()
    else:
        words = list(text)

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    total = pos_count + neg_count
    if total == 0:
        score = 0.5 + np.random.normal(0, 0.05)
    else:
        raw = (pos_count - neg_count) / total
        score = (raw + 1) / 2
        score += np.random.normal(0, 0.03)

    return np.clip(score, 0.01, 0.99)


def classify_sentiment(score):
    if score > 0.6:
        return "positive"
    elif score < 0.4:
        return "negative"
    else:
        return "neutral"


def compute_sentiment_index(group):
    pos = (group["sentiment_label"] == "positive").sum()
    neg = (group["sentiment_label"] == "negative").sum()
    total = len(group)
    if total == 0:
        return 0.0
    return (pos - neg) / total


def analyze_sentiment(input_path="data/weibo_processed.csv",
                      output_results="data/sentiment_results.csv",
                      output_timeseries="data/sentiment_timeseries.csv"):
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} not found. Run preprocess.py first.")
        sys.exit(1)

    print("[INFO] Loading processed data...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[INFO] Loaded {len(df)} posts")

    print("[INFO] Computing sentiment scores...")
    if USE_SNOWNLP:
        print("[INFO] Using SnowNLP for sentiment analysis")
        df["sentiment_score"] = df["cleaned_text"].apply(snownlp_sentiment)
    else:
        print("[INFO] Using dictionary-based sentiment analysis")
        df["sentiment_score"] = df.apply(
            lambda row: dict_sentiment(row["cleaned_text"],
                                       row.get("tokens_str", None)),
            axis=1,
        )

    df["sentiment_label"] = df["sentiment_score"].apply(classify_sentiment)

    df.to_csv(output_results, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved per-post sentiment to {output_results}")

    ts_data = []
    for slice_id in sorted(df["time_slice_id"].unique()):
        group = df[df["time_slice_id"] == slice_id]
        s_t = compute_sentiment_index(group)
        avg_score = group["sentiment_score"].mean()
        pos_count = (group["sentiment_label"] == "positive").sum()
        neg_count = (group["sentiment_label"] == "negative").sum()
        neu_count = (group["sentiment_label"] == "neutral").sum()
        ts_data.append({
            "time_slice_id": slice_id,
            "sentiment_index": s_t,
            "avg_sentiment_score": avg_score,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "total_count": len(group),
        })

    ts_df = pd.DataFrame(ts_data)
    ts_df.to_csv(output_timeseries, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved sentiment time series to {output_timeseries}")

    pos_total = (df["sentiment_label"] == "positive").sum()
    neg_total = (df["sentiment_label"] == "negative").sum()
    neu_total = (df["sentiment_label"] == "neutral").sum()

    print("\n" + "=" * 60)
    print("Sentiment Analysis Summary")
    print("=" * 60)
    print(f"{'Total posts analyzed:':<30}{len(df):>10}")
    print(f"{'Positive:':<30}{pos_total:>10} ({pos_total/len(df)*100:.1f}%)")
    print(f"{'Neutral:':<30}{neu_total:>10} ({neu_total/len(df)*100:.1f}%)")
    print(f"{'Negative:':<30}{neg_total:>10} ({neg_total/len(df)*100:.1f}%)")
    print(f"{'Mean sentiment score:':<30}{df['sentiment_score'].mean():>10.4f}")
    print(f"{'Std sentiment score:':<30}{df['sentiment_score'].std():>10.4f}")
    print(f"{'Time slices:':<30}{len(ts_df):>10}")
    print("=" * 60)

    print("\nSentiment Index by Time Slice (first 10):")
    print(f"{'Slice':>6} {'Index':>8} {'Pos':>6} {'Neu':>6} {'Neg':>6} {'Total':>6}")
    print("-" * 44)
    for _, row in ts_df.head(10).iterrows():
        print(f"{int(row['time_slice_id']):>6} {row['sentiment_index']:>8.4f} "
              f"{int(row['positive_count']):>6} {int(row['neutral_count']):>6} "
              f"{int(row['negative_count']):>6} {int(row['total_count']):>6}")

    annotation_path = "data/manual_annotation.csv"
    if os.path.exists(annotation_path):
        print("\n[INFO] Classification Performance (vs manual annotation):")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        annot = pd.read_csv(annotation_path, encoding="utf-8-sig")
        merged = df.merge(annot[["post_id", "manual_label"]], on="post_id", how="inner")
        if len(merged) > 0:
            label_map = {"positive": 0, "neutral": 1, "negative": 2}
            pred = merged["sentiment_label"].map(label_map).values
            gt = merged["manual_label"].map(label_map).values
            acc = accuracy_score(gt, pred)
            prec = precision_score(gt, pred, average="macro", zero_division=0)
            rec = recall_score(gt, pred, average="macro", zero_division=0)
            f1 = f1_score(gt, pred, average="macro", zero_division=0)
            print(f"{'Annotated samples:':<20}{len(merged)}")
            print(f"{'Accuracy:':<20}{acc:.4f}")
            print(f"{'Precision (macro):':<20}{prec:.4f}")
            print(f"{'Recall (macro):':<20}{rec:.4f}")
            print(f"{'F1-Score (macro):':<20}{f1:.4f}")
    else:
        print("\n[INFO] No manual annotation file found at data/manual_annotation.csv")
        print("[INFO] To evaluate classification performance, provide manual labels.")

    return df, ts_df


if __name__ == "__main__":
    analyze_sentiment()
