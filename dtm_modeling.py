import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

np.random.seed(42)
warnings.filterwarnings("ignore")

try:
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_OK = True
except ImportError:
    print("[WARN] gensim not installed. Run: pip install gensim")
    GENSIM_OK = False

try:
    from gensim.models import CoherenceModel
    COHERENCE_OK = True
except ImportError:
    COHERENCE_OK = False

NUM_TOPICS = 8

TOPIC_NAMES = {
    0: "Disaster Warning",
    1: "Evacuation & Shelter",
    2: "Infrastructure Damage",
    3: "Rescue & Relief",
    4: "Personal Experience",
    5: "Government Response",
    6: "Post-disaster Recovery",
    7: "Media Coverage",
}


def build_corpus_by_slice(df):
    slices = sorted(df["time_slice_id"].unique())
    docs_by_slice = {}
    for s in slices:
        subset = df[df["time_slice_id"] == s]
        token_lists = []
        for t in subset["tokens_str"]:
            if isinstance(t, str) and len(t.strip()) > 0:
                token_lists.append(t.split())
            else:
                token_lists.append([])
        docs_by_slice[s] = token_lists
    return slices, docs_by_slice


def run_sliced_lda(slices, docs_by_slice, num_topics=NUM_TOPICS):
    all_docs = []
    for s in slices:
        all_docs.extend(docs_by_slice[s])

    dictionary = corpora.Dictionary(all_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.7)

    all_corpus = [dictionary.doc2bow(doc) for doc in all_docs]

    print(f"[INFO] Dictionary size: {len(dictionary)} tokens")
    print(f"[INFO] Total documents: {len(all_corpus)}")
    print(f"[INFO] Training LDA with K={num_topics} topics...")

    lda = LdaModel(
        corpus=all_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=15,
        iterations=100,
        random_state=42,
        alpha="auto",
        eta="auto",
    )

    topic_words = {}
    for t in range(num_topics):
        words = lda.show_topic(t, topn=20)
        topic_words[t] = [(w, float(p)) for w, p in words]

    topic_distributions = []
    doc_topics_list = []
    doc_idx = 0

    for s in slices:
        docs = docs_by_slice[s]
        slice_topic_counts = np.zeros(num_topics)
        n_docs = len(docs)

        for doc in docs:
            bow = dictionary.doc2bow(doc)
            if len(bow) == 0:
                doc_topic_dist = np.ones(num_topics) / num_topics
            else:
                topic_dist = lda.get_document_topics(bow, minimum_probability=0.0)
                doc_topic_dist = np.zeros(num_topics)
                for tid, prob in topic_dist:
                    doc_topic_dist[tid] = prob

            slice_topic_counts += doc_topic_dist
            dominant_topic = int(np.argmax(doc_topic_dist))

            doc_topics_list.append({
                "doc_idx": doc_idx,
                "time_slice_id": s,
                "dominant_topic": dominant_topic,
                **{f"topic_{k}": doc_topic_dist[k] for k in range(num_topics)},
            })
            doc_idx += 1

        if n_docs > 0:
            slice_topic_dist = slice_topic_counts / n_docs
        else:
            slice_topic_dist = np.ones(num_topics) / num_topics

        topic_distributions.append({
            "time_slice_id": s,
            **{f"theta_{k}": slice_topic_dist[k] for k in range(num_topics)},
        })

    coherence_score = None
    if COHERENCE_OK:
        try:
            cm = CoherenceModel(
                model=lda,
                texts=all_docs,
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence_score = cm.get_coherence()
        except Exception:
            pass

    return topic_words, topic_distributions, doc_topics_list, coherence_score, lda, dictionary


def auto_name_topics(topic_words):
    keyword_to_topic = {
        "台风": "Disaster Warning",
        "预警": "Disaster Warning",
        "气象": "Disaster Warning",
        "登陆": "Disaster Warning",
        "风力": "Disaster Warning",
        "撤离": "Evacuation & Shelter",
        "转移": "Evacuation & Shelter",
        "安置": "Evacuation & Shelter",
        "停课": "Evacuation & Shelter",
        "封闭": "Evacuation & Shelter",
        "受损": "Infrastructure Damage",
        "停电": "Infrastructure Damage",
        "积水": "Infrastructure Damage",
        "倒塌": "Infrastructure Damage",
        "道路": "Infrastructure Damage",
        "救援": "Rescue & Relief",
        "消防": "Rescue & Relief",
        "武警": "Rescue & Relief",
        "志愿者": "Rescue & Relief",
        "抢修": "Rescue & Relief",
        "害怕": "Personal Experience",
        "恐惧": "Personal Experience",
        "进水": "Personal Experience",
        "祈祷": "Personal Experience",
        "平安": "Personal Experience",
        "政府": "Government Response",
        "会议": "Government Response",
        "部署": "Government Response",
        "应急": "Government Response",
        "通知": "Government Response",
        "恢复": "Post-disaster Recovery",
        "重建": "Post-disaster Recovery",
        "清理": "Post-disaster Recovery",
        "理赔": "Post-disaster Recovery",
        "复盘": "Post-disaster Recovery",
        "记者": "Media Coverage",
        "报道": "Media Coverage",
        "直播": "Media Coverage",
        "统计": "Media Coverage",
        "央视": "Media Coverage",
    }

    names = {}
    used_names = set()
    for tid, words in topic_words.items():
        scores = defaultdict(float)
        for word, prob in words[:10]:
            if word in keyword_to_topic:
                name = keyword_to_topic[word]
                scores[name] += prob

        if scores:
            sorted_names = sorted(scores.items(), key=lambda x: -x[1])
            for name, _ in sorted_names:
                if name not in used_names:
                    names[tid] = name
                    used_names.add(name)
                    break

        if tid not in names:
            names[tid] = TOPIC_NAMES.get(tid, f"Topic {tid}")

    return names


def run_dtm(input_path="data/weibo_processed.csv",
            output_topic_dist="data/topic_distributions.csv",
            output_topic_words="data/topic_words.csv",
            output_doc_topics="data/doc_topics.csv"):
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} not found. Run preprocess.py first.")
        sys.exit(1)

    if not GENSIM_OK:
        print("[ERROR] gensim is required for DTM modeling.")
        sys.exit(1)

    print("[INFO] Loading processed data...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[INFO] Loaded {len(df)} posts across {df['time_slice_id'].nunique()} time slices")

    slices, docs_by_slice = build_corpus_by_slice(df)

    topic_words, topic_distributions, doc_topics_list, coherence_score, lda, dictionary = \
        run_sliced_lda(slices, docs_by_slice, num_topics=NUM_TOPICS)

    topic_names = auto_name_topics(topic_words)

    td_df = pd.DataFrame(topic_distributions)
    td_df.to_csv(output_topic_dist, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved topic distributions to {output_topic_dist}")

    tw_records = []
    for tid, words in topic_words.items():
        for rank, (word, prob) in enumerate(words):
            tw_records.append({
                "topic_id": tid,
                "topic_name": topic_names.get(tid, f"Topic {tid}"),
                "rank": rank + 1,
                "word": word,
                "probability": prob,
            })
    tw_df = pd.DataFrame(tw_records)
    tw_df.to_csv(output_topic_words, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved topic words to {output_topic_words}")

    dt_df = pd.DataFrame(doc_topics_list)
    dt_df.to_csv(output_doc_topics, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved document topics to {output_doc_topics}")

    print("\n" + "=" * 60)
    print("Dynamic Topic Modeling Summary")
    print("=" * 60)
    print(f"{'Number of topics:':<30}{NUM_TOPICS:>10}")
    print(f"{'Time slices:':<30}{len(slices):>10}")
    print(f"{'Dictionary size:':<30}{len(dictionary):>10}")
    if coherence_score is not None:
        print(f"{'Coherence (c_v):':<30}{coherence_score:>10.4f}")
    print("=" * 60)

    print("\nTopic Keywords:")
    print("-" * 60)
    for tid in range(NUM_TOPICS):
        name = topic_names.get(tid, f"Topic {tid}")
        top5 = ", ".join([w for w, _ in topic_words[tid][:5]])
        print(f"  Topic {tid} [{name}]: {top5}")

    print("\nTopic Strength by Time Slice (theta_k):")
    print(f"{'Slice':>6}", end="")
    for k in range(NUM_TOPICS):
        print(f"  T{k:>1}", end="")
    print()
    print("-" * (6 + NUM_TOPICS * 5))
    for _, row in td_df.iterrows():
        print(f"{int(row['time_slice_id']):>6}", end="")
        for k in range(NUM_TOPICS):
            print(f" {row[f'theta_{k}']:.2f}", end="")
        print()

    return td_df, tw_df, dt_df, topic_names


if __name__ == "__main__":
    run_dtm()
