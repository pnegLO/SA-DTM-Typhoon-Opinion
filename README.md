# SA-DTM: Sentiment-Aware Dynamic Topic Model for Disaster Opinion Analysis

A fusion framework for analyzing public opinion evolution during sudden natural disasters on social media, integrating sentiment analysis, dynamic topic modeling, and LSTM-based prediction.

## Overview

This project implements the **SA-DTM-LSTM** pipeline for typhoon-related Weibo (microblog) data analysis:

1. **Data Collection** (`collect_weibo.py`) — Collect Weibo posts via API
2. **Preprocessing** (`preprocess.py`) — Text cleaning, tokenization (jieba), 3-hour time slicing
3. **Sentiment Analysis** (`sentiment_analysis.py`) — SnowNLP-based sentiment classification (positive/neutral/negative)
4. **Dynamic Topic Modeling** (`dtm_modeling.py`) — Time-sliced LDA with K=8 topics
5. **Feature Fusion** (`feature_fusion.py`) — Topic-Sentiment Matrix (TSM) + EWM smoothing + PCA reduction
6. **Prediction** (`lstm_prediction.py`) — Attention-LSTM with ablation experiments (5 models + 4 ablation variants)
7. **Visualization** (`visualize.py`) — 8 publication-quality figures

## Key Innovation

- **TSM (Topic-Sentiment Matrix)**: Captures within-topic sentiment variation, achieving Spearman ρ=0.639 (p<0.001) — far exceeding single-dimension features
- **EWM + PCA Pipeline**: Exponential weighted moving average smoothing + PCA dimensionality reduction (17D → 7D, 85.7% variance explained)
- **Temporal Attention LSTM**: Self-adaptive weighting of historical time steps for interpretable prediction

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| ARIMA | 0.0824 | 0.0970 |
| GRU | 0.0759 | 0.0948 |
| SA-LSTM | 0.0999 | 0.1185 |
| DTM-LSTM | 0.0645 | 0.0794 |
| **SA-DTM-LSTM** | **0.0637** | **0.0796** |

SA-DTM-LSTM reduces MAE by **22.7%** compared to ARIMA baseline.

## Quick Start

```bash
pip install numpy pandas jieba snownlp gensim torch statsmodels matplotlib seaborn scikit-learn

python run_all.py
```

## Project Structure

```
├── collect_weibo.py        # Weibo data collection
├── preprocess.py           # Text preprocessing & time slicing
├── sentiment_analysis.py   # SnowNLP sentiment classification
├── dtm_modeling.py         # Dynamic topic modeling
├── feature_fusion.py       # TSM + EWM + PCA feature engineering
├── lstm_prediction.py      # Attention-LSTM prediction & ablation
├── visualize.py            # Figure generation
├── run_all.py              # One-click pipeline
├── data/                   # Input data & intermediate results
└── figures/                # Output figures (PNG + PDF)
```

## Requirements

- Python 3.9+
- PyTorch 1.13+
- gensim 4.3+
- jieba, snownlp, statsmodels, scikit-learn, matplotlib, seaborn

## Citation

If you use this code in your research, please cite:

```
@mastersthesis{sa_dtm_2025,
  title={基于SA-DTM模型的突发自然灾害舆情演化分析},
  author={},
  year={2025},
  school={}
}
```

## Related Publications & Research Outputs

This repository accompanies an academic thesis. **Complete research outputs including the full thesis, experimental data, and trained models are available for academic collaboration and licensing.**

For inquiries regarding:
- Full thesis document
- Experimental datasets
- Trained model weights
- Academic collaboration

**Contact**: pj206323@gmail.com

## License

MIT License
