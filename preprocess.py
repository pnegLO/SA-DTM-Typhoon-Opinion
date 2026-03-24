import os
import re
import sys
import time
import numpy as np
import pandas as pd

np.random.seed(42)

try:
    import jieba
except ImportError:
    print("[ERROR] jieba not installed. Run: pip install jieba")
    sys.exit(1)


STOPWORDS_DEFAULT = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 里 为 什么 吗 吧 呢 啊 哦 嗯 呀 哈 哟 嘛 呗 "
    "被 把 让 给 向 从 对 于 但 而 或 如果 虽然 因为 所以 可以 已经 还 又 再 "
    "这个 那个 这些 那些 之 与 及 等 来 去 过 中 大 小 多 少 第 次 个 年 月 日 时 "
    "分 秒 号 更 最 其 该 某 每 各 所有 以及 以上 以下 之间 之后 之前 "
    "http https com cn www 的话 然后 其实 可能 应该 觉得 知道 感觉 希望 "
    "RT 转发 微博 回复 评论".split()
)

AD_KEYWORDS = [
    "优惠", "促销", "折扣", "秒杀", "抢购", "代购", "加微信", "加V",
    "免费领", "点击链接", "扫码", "下单", "淘宝", "拼多多", "京东",
    "砍价", "红包", "返现", "佣金", "招代理",
]


def load_stopwords():
    stopwords = STOPWORDS_DEFAULT.copy()
    stopwords_paths = [
        "data/stopwords.txt",
        os.path.join(os.path.dirname(__file__), "data", "stopwords.txt"),
    ]
    for path in stopwords_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stopwords.add(line.strip())
            break
    return stopwords


def clean_text(text):
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"@[\w\u4e00-\u9fff]+", "", text)
    text = re.sub(r"#.*?#", "", text)
    text = re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                  r"\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
                  r"\U00002702-\U000027B0]+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\w\s，。！？、；：""''（）]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_ad(text):
    for kw in AD_KEYWORDS:
        if kw in text:
            return True
    return False


def tokenize(text, stopwords):
    words = jieba.lcut(text)
    tokens = [w for w in words if len(w) > 1 and w not in stopwords and not w.isdigit()]
    return tokens


def assign_time_slice(df, freq_hours=3):
    time_col = "publish_time" if "publish_time" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[time_col])
    min_time = df["timestamp"].min().floor("D")
    df["hours_since_start"] = (df["timestamp"] - min_time).dt.total_seconds() / 3600
    df["time_slice_id"] = (df["hours_since_start"] // freq_hours).astype(int)
    return df


def preprocess(input_path="data/weibo_typhoon_raw.csv", output_path="data/weibo_processed.csv"):
    if not os.path.exists(input_path):
        print(f"[ERROR] Raw data not found at {input_path}")
        print("[INFO] Please run collect_weibo.py first to collect data.")
        sys.exit(1)

    print(f"[INFO] Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[INFO] Raw data: {len(df)} posts")

    original_count = len(df)

    df = df.drop_duplicates(subset=["text"], keep="first")
    dedup_count = len(df)
    print(f"[INFO] After deduplication: {dedup_count} posts (removed {original_count - dedup_count})")

    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 5].reset_index(drop=True)
    clean_count = len(df)
    print(f"[INFO] After cleaning short texts: {clean_count} posts")

    ad_mask = df["cleaned_text"].apply(is_ad)
    df = df[~ad_mask].reset_index(drop=True)
    no_ad_count = len(df)
    print(f"[INFO] After removing ads: {no_ad_count} posts (removed {clean_count - no_ad_count})")

    stopwords = load_stopwords()
    df["tokens"] = df["cleaned_text"].apply(lambda x: tokenize(x, stopwords))
    df = df[df["tokens"].apply(len) >= 2].reset_index(drop=True)

    df = assign_time_slice(df, freq_hours=3)

    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    final_count = len(df)
    n_slices = df["time_slice_id"].nunique()

    print("\n" + "=" * 60)
    print("Data Preprocessing Summary")
    print("=" * 60)
    print(f"{'Raw posts:':<30}{original_count:>10}")
    print(f"{'After deduplication:':<30}{dedup_count:>10}")
    print(f"{'After cleaning:':<30}{clean_count:>10}")
    print(f"{'After removing ads:':<30}{no_ad_count:>10}")
    print(f"{'Final processed posts:':<30}{final_count:>10}")
    print(f"{'Removal rate:':<30}{(1 - final_count/original_count)*100:>9.1f}%")
    print(f"{'Time slices (6h each):':<30}{n_slices:>10}")
    print(f"{'Avg posts per slice:':<30}{final_count/n_slices:>10.1f}")
    print("=" * 60)

    slice_stats = df.groupby("time_slice_id").size()
    print(f"\nPosts per time slice: min={slice_stats.min()}, max={slice_stats.max()}, "
          f"mean={slice_stats.mean():.1f}, std={slice_stats.std():.1f}")

    return df


if __name__ == "__main__":
    preprocess()
