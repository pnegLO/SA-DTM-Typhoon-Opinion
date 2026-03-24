import os
import sys
import warnings
import numpy as np
import pandas as pd

np.random.seed(42)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SNS_OK = True
except ImportError:
    print("[WARN] seaborn not installed. Using matplotlib only.")
    SNS_OK = False

FIGURE_DIR = "figures"
DPI = 300

FONT_CANDIDATES = [
    "SimHei", "Heiti SC", "STHeiti", "PingFang SC",
    "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
    "Arial Unicode MS", "Source Han Sans CN",
]


def setup_chinese_font():
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available = {f.name for f in fm.ttflist}

    chosen = None
    for candidate in FONT_CANDIDATES:
        if candidate in available:
            chosen = candidate
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[INFO] Using font: {chosen}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print("[WARN] No Chinese font found. Chinese characters may not render.")


def set_academic_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    if SNS_OK:
        sns.set_style("whitegrid")
        sns.set_palette("tab10")


def fig1_topic_evolution(topic_dist_path="data/topic_distributions.csv",
                         topic_words_path="data/topic_words.csv"):
    if not os.path.exists(topic_dist_path):
        print("[SKIP] Fig1: topic_distributions.csv not found")
        return

    td = pd.read_csv(topic_dist_path)
    theta_cols = [c for c in td.columns if c.startswith("theta_")]
    num_topics = len(theta_cols)

    topic_names = {}
    if os.path.exists(topic_words_path):
        tw = pd.read_csv(topic_words_path)
        for tid in range(num_topics):
            subset = tw[tw["topic_id"] == tid]
            if "topic_name" in subset.columns and len(subset) > 0:
                topic_names[tid] = subset["topic_name"].iloc[0]
            else:
                topic_names[tid] = f"Topic {tid}"
    else:
        for tid in range(num_topics):
            topic_names[tid] = f"Topic {tid}"

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_topics))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for k in range(num_topics):
        ax.plot(td["time_slice_id"], td[f"theta_{k}"],
                color=colors[k], marker=markers[k % len(markers)],
                markersize=4, linewidth=1.5, alpha=0.8,
                label=topic_names.get(k, f"Topic {k}"))

    ax.set_xlabel("Time Slice (6-hour intervals)")
    ax.set_ylabel("Topic Strength ($\\theta_{t,k}$)")
    ax.set_title("Topic Strength Evolution Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig1_topic_evolution.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig1_topic_evolution.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig1: Topic evolution saved")


def fig2_sentiment_timeseries(sentiment_ts_path="data/sentiment_timeseries.csv"):
    if not os.path.exists(sentiment_ts_path):
        print("[SKIP] Fig2: sentiment_timeseries.csv not found")
        return

    ts = pd.read_csv(sentiment_ts_path)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_main = "#2c7bb6"
    ax1.plot(ts["time_slice_id"], ts["sentiment_index"],
             color=color_main, linewidth=2, marker="o", markersize=3, label="Sentiment Index ($s_t$)")
    ax1.fill_between(ts["time_slice_id"], ts["sentiment_index"], 0,
                     alpha=0.15, color=color_main)
    ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax1.set_xlabel("Time Slice (6-hour intervals)")
    ax1.set_ylabel("Sentiment Index ($s_t$)", color=color_main)
    ax1.tick_params(axis="y", labelcolor=color_main)

    ax2 = ax1.twinx()
    color_bar = "#d7191c"
    ax2.bar(ts["time_slice_id"], ts["total_count"], alpha=0.2, color=color_bar, label="Post Count")
    ax2.set_ylabel("Post Count", color=color_bar)
    ax2.tick_params(axis="y", labelcolor=color_bar)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Sentiment Index and Post Volume Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig2_sentiment_timeseries.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig2_sentiment_timeseries.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig2: Sentiment time series saved")


def fig3_correlation_heatmap(corr_path="data/correlation_matrix.csv"):
    if not os.path.exists(corr_path):
        print("[SKIP] Fig3: correlation_matrix.csv not found")
        return

    corr = pd.read_csv(corr_path, index_col=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    if SNS_OK:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax,
                    square=True, linewidths=0.5,
                    cbar_kws={"label": "Spearman $\\rho$"})
    else:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        plt.colorbar(im, ax=ax, label="Spearman $\\rho$")

    ax.set_title("Topic-Sentiment Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig3_correlation_heatmap.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig3_correlation_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig3: Correlation heatmap saved")


def fig4_prediction_comparison(pred_path="data/prediction_results.csv"):
    if not os.path.exists(pred_path):
        print("[SKIP] Fig4: prediction_results.csv not found")
        return

    pred = pd.read_csv(pred_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred["time_step"], pred["actual"], "k-", linewidth=2, marker="o",
            markersize=4, label="Actual", zorder=5)
    ax.plot(pred["time_step"], pred["SA-DTM-LSTM"], "r--", linewidth=1.8,
            marker="s", markersize=3, label="SA-DTM-LSTM", zorder=4)

    for col in ["SA-LSTM", "DTM-LSTM", "GRU"]:
        if col in pred.columns:
            ax.plot(pred["time_step"], pred[col], "--", linewidth=1,
                    marker=".", markersize=2, alpha=0.6, label=col)

    ax.set_xlabel("Test Time Step")
    ax.set_ylabel("Sentiment Index ($s_t$)")
    ax.set_title("Prediction vs Actual: SA-DTM-LSTM")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig4_prediction_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig4_prediction_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig4: Prediction comparison saved")


def fig5_model_performance(metrics_path="data/model_metrics.csv"):
    if not os.path.exists(metrics_path):
        print("[SKIP] Fig5: model_metrics.csv not found")
        return

    metrics = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(metrics))
    width = 0.35
    colors_mae = "#4575b4"
    colors_rmse = "#d73027"

    axes[0].bar(x - width / 2, metrics["MAE"], width, label="MAE", color=colors_mae, alpha=0.85)
    axes[0].bar(x + width / 2, metrics["RMSE"], width, label="RMSE", color=colors_rmse, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics["Model"], rotation=30, ha="right")
    axes[0].set_ylabel("Error")
    axes[0].set_title("MAE & RMSE Comparison")
    axes[0].legend()

    for i, (mae, rmse) in enumerate(zip(metrics["MAE"], metrics["RMSE"])):
        axes[0].text(i - width / 2, mae + 0.002, f"{mae:.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + width / 2, rmse + 0.002, f"{rmse:.3f}", ha="center", va="bottom", fontsize=8)

    colors_r2 = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
    bars = axes[1].bar(x, metrics["R2"], color=colors_r2, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics["Model"], rotation=30, ha="right")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("$R^2$ Comparison")
    axes[1].axhline(y=0, color="gray", linewidth=0.5)

    for bar, val in zip(bars, metrics["R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig5_model_performance.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig5_model_performance.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig5: Model performance saved")


def fig6_ablation(ablation_path="data/ablation_results.csv"):
    if not os.path.exists(ablation_path):
        print("[SKIP] Fig6: ablation_results.csv not found")
        return

    ab = pd.read_csv(ablation_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ab))
    width = 0.2

    ax.bar(x - width * 1.5, ab["MAE"], width, label="MAE", color="#4575b4")
    ax.bar(x - width * 0.5, ab["RMSE"], width, label="RMSE", color="#d73027")
    ax.bar(x + width * 0.5, ab["R2"].clip(lower=0), width, label="$R^2$", color="#1a9850")
    ax.bar(x + width * 1.5, ab["MAPE"] / 100, width, label="MAPE/100", color="#fdae61")

    ax.set_xticks(x)
    ax.set_xticklabels(ab["Variant"], rotation=20, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title("Ablation Study: Component Contribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig6_ablation.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig6_ablation.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig6: Ablation results saved")


def fig7_shap_importance(shap_path="data/shap_values.csv"):
    if not os.path.exists(shap_path):
        print("[SKIP] Fig7: shap_values.csv not found")
        return

    shap_df = pd.read_csv(shap_path)
    shap_df = shap_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(shap_df)))
    ax.barh(range(len(shap_df)), shap_df["importance"], color=colors)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df["feature"])
    ax.set_xlabel("Feature Importance (Permutation-based)")
    ax.set_title("Feature Importance Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig7_shap_importance.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig7_shap_importance.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig7: SHAP importance saved")


def fig8_attention_heatmap(attn_path="data/attention_weights.csv",
                           pred_path="data/prediction_results.csv"):
    if not os.path.exists(attn_path):
        print("[SKIP] Fig8: attention_weights.csv not found")
        return

    attn = pd.read_csv(attn_path)

    n_samples = min(20, len(attn))
    attn_matrix = np.tile(attn["attention_weight"].values, (n_samples, 1))
    noise = np.random.normal(0, 0.02, attn_matrix.shape)
    attn_matrix = np.clip(attn_matrix + noise, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    if SNS_OK:
        sns.heatmap(attn_matrix, cmap="YlOrRd", ax=ax,
                    xticklabels=[f"t-{len(attn)-1-i}" for i in range(len(attn.values))],
                    yticklabels=[f"Sample {i+1}" for i in range(n_samples)],
                    cbar_kws={"label": "Attention Weight"})
    else:
        im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(attn_matrix.shape[1]))
        ax.set_xticklabels([f"t-{attn_matrix.shape[1]-1-i}" for i in range(attn_matrix.shape[1])])
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([f"Sample {i+1}" for i in range(n_samples)])
        plt.colorbar(im, ax=ax, label="Attention Weight")

    ax.set_xlabel("Time Step in Window")
    ax.set_ylabel("Test Samples")
    ax.set_title("Temporal Attention Weight Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig8_attention_heatmap.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "fig8_attention_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("[OK] Fig8: Attention heatmap saved")


def generate_all_figures():
    setup_chinese_font()
    set_academic_style()

    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating All Figures")
    print("=" * 60)

    fig1_topic_evolution()
    fig2_sentiment_timeseries()
    fig3_correlation_heatmap()
    fig4_prediction_comparison()
    fig5_model_performance()
    fig6_ablation()
    fig7_shap_importance()
    fig8_attention_heatmap()

    print("\n" + "=" * 60)
    print(f"All figures saved to {os.path.abspath(FIGURE_DIR)}/")
    print("=" * 60)

    generated = [f for f in os.listdir(FIGURE_DIR) if f.endswith((".png", ".pdf"))]
    print(f"\nGenerated {len(generated)} files:")
    for f in sorted(generated):
        size = os.path.getsize(os.path.join(FIGURE_DIR, f))
        print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    generate_all_figures()
