# SA-DTM：突发自然灾害社交媒体舆情主题-情感协同演化分析与预测

融合情感分析、动态主题模型与注意力LSTM的灾害微博舆情分析框架。

## 项目简介

本项目针对突发自然灾害（以台风为案例）的微博舆情数据，构建 **SA-DTM-LSTM** 融合分析与预测管线：

- 采集微博平台灾害相关文本数据
- 通过SnowNLP执行中文情感分类，通过DTM追踪主题动态演化
- 构建**主题-情感交互矩阵（TSM）**刻画主题内部情感分化
- 经EWM平滑与PCA降维后输入带时序注意力的LSTM网络，预测舆情情感走势
- 消融实验验证各组件的增量贡献

## 技术创新

**TSM主题-情感交互矩阵**：对每个时间片内的每个主题，计算该主题下文档的加权平均情感得分，生成主题-情感耦合信号。实验表明TSM交互特征与情感指数的Spearman相关系数达ρ=0.639（p<0.001），远超单一主题强度的相关水平。

**EWM+PCA特征工程管线**：指数加权移动平均平滑抑制短周期噪声，主成分分析将17维原始特征压缩至7维（解释方差85.7%），有效缓解小样本高维输入的过拟合问题。

**时序注意力机制**：在LSTM输出层前嵌入注意力模块，自适应聚焦对当前预测贡献最大的历史时间步，增强模型对舆情转折点的响应能力与可解释性。

## 实验结果

| 模型 | MAE | RMSE |
|------|-----|------|
| ARIMA | 0.0824 | 0.0970 |
| GRU | 0.0759 | 0.0948 |
| SA-LSTM | 0.0999 | 0.1185 |
| DTM-LSTM | 0.0645 | 0.0794 |
| **SA-DTM-LSTM** | **0.0637** | **0.0796** |

SA-DTM-LSTM较ARIMA基线MAE降低**22.7%**，消融实验验证TSM交互特征和注意力机制均有不可替代的增量贡献。

## 文件说明

| 文件 | 功能 |
|------|------|
| `collect_weibo.py` | 微博数据采集（API接口+爬虫） |
| `preprocess.py` | 文本预处理：去重、分词、3小时时间片划分 |
| `sentiment_analysis.py` | SnowNLP情感分类（正面/中性/负面） |
| `dtm_modeling.py` | 动态主题模型（K=8个主题） |
| `feature_fusion.py` | TSM矩阵构建 + EWM平滑 + PCA降维 |
| `lstm_prediction.py` | Attention-LSTM预测 + 五组模型对比 + 消融实验 |
| `visualize.py` | 生成8张论文级图表 |
| `run_all.py` | 一键运行全流程 |

## 快速开始

```bash
pip install numpy pandas jieba snownlp gensim torch statsmodels matplotlib seaborn scikit-learn

python run_all.py
```

运行后自动在 `data/` 目录生成全部中间结果，在 `figures/` 目录生成8张图表（PNG+PDF）。

## 运行环境

- Python 3.9+
- PyTorch 1.13+
- gensim 4.3+
- jieba、snownlp、statsmodels、scikit-learn、matplotlib、seaborn

## 相关论文与科研成果

本仓库为学位论文《基于SA-DTM模型的突发自然灾害舆情演化分析》的配套代码。

**完整科研成果（含论文全文、实验数据集、训练模型权重）可提供学术合作与授权使用。**

如有以下需求欢迎联系：
- 论文全文获取
- 实验数据集与标注数据
- 模型权重与复现指导
- 学术合作与定制开发

📧 **联系邮箱**：pj206323@gmail.com

## 许可证

MIT License

---

# SA-DTM: Sentiment-Aware Dynamic Topic Model for Disaster Opinion Analysis

A fusion framework integrating sentiment analysis, dynamic topic modeling, and Attention-LSTM for analyzing public opinion evolution during sudden natural disasters on social media.

## Highlights

- **TSM (Topic-Sentiment Matrix)**: Within-topic sentiment variation capture, Spearman ρ=0.639 (p<0.001)
- **EWM + PCA Pipeline**: 17D → 7D feature compression with 85.7% variance explained
- **Attention-LSTM**: Temporal attention for interpretable prediction, 22.7% MAE reduction over ARIMA

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| ARIMA | 0.0824 | 0.0970 |
| GRU | 0.0759 | 0.0948 |
| SA-LSTM | 0.0999 | 0.1185 |
| DTM-LSTM | 0.0645 | 0.0794 |
| **SA-DTM-LSTM** | **0.0637** | **0.0796** |

## Quick Start

```bash
pip install numpy pandas jieba snownlp gensim torch statsmodels matplotlib seaborn scikit-learn
python run_all.py
```

## Research Outputs

This repository accompanies the master's thesis *"Analysis of Public Opinion Evolution in Sudden Natural Disasters Based on SA-DTM Model"*.

**Full research outputs (thesis, datasets, trained models) available for academic collaboration and licensing.**

📧 **Contact**: pj206323@gmail.com

## License

MIT License
