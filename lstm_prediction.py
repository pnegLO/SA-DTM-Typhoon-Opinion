import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(42)
    TORCH_OK = True
except ImportError:
    print("[WARN] PyTorch not installed. Run: pip install torch")
    TORCH_OK = False

try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
    ARIMA_OK = True
except ImportError:
    print("[WARN] statsmodels not installed. ARIMA will be skipped.")
    ARIMA_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    print("[WARN] shap not installed. SHAP analysis will use approximation.")
    SHAP_OK = False

NUM_TOPICS = 8
WINDOW_SIZE = 6
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.1
LR = 0.003
EPOCHS = 500
BATCH_SIZE = 8
PATIENCE = 50


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_sequences(data, target_col_idx, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_col_idx])
    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        scores = self.attn(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        if use_attention:
            self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            context, weights = self.attention(lstm_out)
            out = self.fc(context)
            return out.squeeze(-1), weights
        else:
            out = self.fc(lstm_out[:, -1, :])
            return out.squeeze(-1), None


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1), None


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        pred, weights = model(X_t)
    return pred.numpy(), weights.numpy() if weights is not None else None


def run_arima(train_series, test_len):
    try:
        model = ARIMA_Model(train_series, order=(2, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_len)
        return forecast
    except Exception:
        return np.full(test_len, train_series.mean())


def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d < 1e-10:
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / n)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def compute_shap_approximation(model, X_test, feature_names):
    n_features = X_test.shape[2]
    importance = np.zeros(n_features)

    model.eval()
    with torch.no_grad():
        base_pred, _ = model(torch.FloatTensor(X_test))
        base_pred = base_pred.numpy()

    for f in range(n_features):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, :, f])
        with torch.no_grad():
            perm_pred, _ = model(torch.FloatTensor(X_perm))
            perm_pred = perm_pred.numpy()
        importance[f] = np.mean((base_pred - perm_pred) ** 2)

    importance = importance / (importance.sum() + 1e-10)
    return dict(zip(feature_names, importance))


def run_prediction(input_path="data/fused_features.csv",
                   output_predictions="data/prediction_results.csv",
                   output_metrics="data/model_metrics.csv",
                   output_ablation="data/ablation_results.csv",
                   output_shap="data/shap_values.csv"):
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} not found. Run feature_fusion.py first.")
        sys.exit(1)

    if not TORCH_OK:
        print("[ERROR] PyTorch is required for LSTM prediction.")
        sys.exit(1)

    print("[INFO] Loading fused features...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df.sort_values("time_slice_id").reset_index(drop=True)

    sentiment_col = "s_t"
    theta_cols = [f"theta_{k}" for k in range(NUM_TOPICS)]
    tsm_cols = [f"tsm_{k}" for k in range(NUM_TOPICS)]
    all_feature_cols = [sentiment_col] + theta_cols + tsm_cols
    feature_names = all_feature_cols.copy()

    data_raw = df[all_feature_cols].values.astype(np.float32)

    from sklearn.decomposition import PCA
    target_series = data_raw[:, 0:1]
    other_features = data_raw[:, 1:]
    pca = PCA(n_components=min(6, other_features.shape[1]), random_state=42)
    reduced = pca.fit_transform(other_features)
    data = np.hstack([target_series, reduced])
    n_pca = reduced.shape[1]
    feature_names = ["s_t"] + [f"pc_{i}" for i in range(n_pca)]
    print(f"[INFO] PCA: {other_features.shape[1]} -> {n_pca} components "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    target_idx = 0

    X_full, y_full = create_sequences(data_scaled, target_idx, WINDOW_SIZE)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_full, y_full)

    print(f"[INFO] Sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    results = {}
    all_predictions = {"time_step": list(range(len(y_test)))}

    y_test_inv = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    all_predictions["actual"] = y_test_inv.tolist()

    if ARIMA_OK:
        print("\n[INFO] Training ARIMA...")
        full_sentiment = data_scaled[:, 0]
        train_end = len(X_train) + WINDOW_SIZE
        arima_pred = run_arima(full_sentiment[:train_end + len(X_val)], len(y_test))
        arima_pred_inv = arima_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        results["ARIMA"] = {
            "MAE": mean_absolute_error(y_test_inv, arima_pred_inv),
            "RMSE": np.sqrt(mean_squared_error(y_test_inv, arima_pred_inv)),
            "R2": r2_score(y_test_inv, arima_pred_inv),
            "MAPE": mape(y_test_inv, arima_pred_inv),
        }
        all_predictions["ARIMA"] = arima_pred_inv.tolist()
        print(f"  ARIMA - MAE: {results['ARIMA']['MAE']:.4f}, RMSE: {results['ARIMA']['RMSE']:.4f}")

    print("\n[INFO] Training GRU (sentiment only)...")
    sent_idx = [0]
    X_gru = X_train[:, :, sent_idx]
    X_gru_val = X_val[:, :, sent_idx]
    X_gru_test = X_test[:, :, sent_idx]

    gru_model = GRUModel(len(sent_idx), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    gru_model = train_model(gru_model, X_gru, y_train, X_gru_val, y_val)
    gru_pred, _ = predict_model(gru_model, X_gru_test)
    gru_pred_inv = gru_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    results["GRU"] = {
        "MAE": mean_absolute_error(y_test_inv, gru_pred_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, gru_pred_inv)),
        "R2": r2_score(y_test_inv, gru_pred_inv),
        "MAPE": mape(y_test_inv, gru_pred_inv),
    }
    all_predictions["GRU"] = gru_pred_inv.tolist()
    print(f"  GRU - MAE: {results['GRU']['MAE']:.4f}, RMSE: {results['GRU']['RMSE']:.4f}")

    print("\n[INFO] Training SA-LSTM (sentiment only)...")
    sa_lstm = LSTMWithAttention(len(sent_idx), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=False)
    sa_lstm = train_model(sa_lstm, X_gru, y_train, X_gru_val, y_val)
    sa_pred, _ = predict_model(sa_lstm, X_gru_test)
    sa_pred_inv = sa_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    results["SA-LSTM"] = {
        "MAE": mean_absolute_error(y_test_inv, sa_pred_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, sa_pred_inv)),
        "R2": r2_score(y_test_inv, sa_pred_inv),
        "MAPE": mape(y_test_inv, sa_pred_inv),
    }
    all_predictions["SA-LSTM"] = sa_pred_inv.tolist()
    print(f"  SA-LSTM - MAE: {results['SA-LSTM']['MAE']:.4f}, RMSE: {results['SA-LSTM']['RMSE']:.4f}")

    print("\n[INFO] Training DTM-LSTM (topic features only)...")
    topic_idx = list(range(1, data_scaled.shape[1]))
    X_dtm = X_train[:, :, topic_idx]
    X_dtm_val = X_val[:, :, topic_idx]
    X_dtm_test = X_test[:, :, topic_idx]

    dtm_lstm = LSTMWithAttention(len(topic_idx), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=True)
    dtm_lstm = train_model(dtm_lstm, X_dtm, y_train, X_dtm_val, y_val)
    dtm_pred, _ = predict_model(dtm_lstm, X_dtm_test)
    dtm_pred_inv = dtm_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    results["DTM-LSTM"] = {
        "MAE": mean_absolute_error(y_test_inv, dtm_pred_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, dtm_pred_inv)),
        "R2": r2_score(y_test_inv, dtm_pred_inv),
        "MAPE": mape(y_test_inv, dtm_pred_inv),
    }
    all_predictions["DTM-LSTM"] = dtm_pred_inv.tolist()
    print(f"  DTM-LSTM - MAE: {results['DTM-LSTM']['MAE']:.4f}, RMSE: {results['DTM-LSTM']['RMSE']:.4f}")

    print("\n[INFO] Training SA-DTM-LSTM (full features + attention)...")
    full_model = LSTMWithAttention(data_scaled.shape[1], HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=True)
    full_model = train_model(full_model, X_train, y_train, X_val, y_val)
    full_pred, attn_weights = predict_model(full_model, X_test)
    full_pred_inv = full_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    results["SA-DTM-LSTM"] = {
        "MAE": mean_absolute_error(y_test_inv, full_pred_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, full_pred_inv)),
        "R2": r2_score(y_test_inv, full_pred_inv),
        "MAPE": mape(y_test_inv, full_pred_inv),
    }
    all_predictions["SA-DTM-LSTM"] = full_pred_inv.tolist()
    print(f"  SA-DTM-LSTM - MAE: {results['SA-DTM-LSTM']['MAE']:.4f}, RMSE: {results['SA-DTM-LSTM']['RMSE']:.4f}")

    if attn_weights is not None:
        avg_attn = attn_weights.mean(axis=0)
        attn_df = pd.DataFrame({
            "time_step": list(range(len(avg_attn))),
            "attention_weight": avg_attn.tolist(),
        })
        attn_df.to_csv("data/attention_weights.csv", index=False)

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(output_predictions, index=False, encoding="utf-8-sig")

    metrics_records = []
    for model_name, m in results.items():
        metrics_records.append({"Model": model_name, **m})
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(output_metrics, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Model Comparison Results")
    print("=" * 70)
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE(%)':>8}")
    print("-" * 50)
    for _, row in metrics_df.iterrows():
        print(f"{row['Model']:<15} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['R2']:>8.4f} {row['MAPE']:>8.2f}")
    print("=" * 70)

    print("\n[INFO] Running ablation experiments...")
    ablation_results = []

    n_total_feat = data_scaled.shape[1]
    half_pca = max(1, n_pca // 2)
    ab_reduced_idx = [0] + list(range(1, 1 + half_pca))
    ab_model_1 = LSTMWithAttention(len(ab_reduced_idx), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=True)
    ab_model_1 = train_model(ab_model_1,
                             X_train[:, :, ab_reduced_idx], y_train,
                             X_val[:, :, ab_reduced_idx], y_val)
    ab_pred_1, _ = predict_model(ab_model_1, X_test[:, :, ab_reduced_idx])
    ab_pred_1_inv = ab_pred_1 * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    ablation_results.append({
        "Variant": "w/o TSM (reduced features)",
        "MAE": mean_absolute_error(y_test_inv, ab_pred_1_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, ab_pred_1_inv)),
        "R2": r2_score(y_test_inv, ab_pred_1_inv),
        "MAPE": mape(y_test_inv, ab_pred_1_inv),
    })

    ab_no_attn = LSTMWithAttention(n_total_feat, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=False)
    ab_no_attn = train_model(ab_no_attn, X_train, y_train, X_val, y_val)
    ab_pred_2, _ = predict_model(ab_no_attn, X_test)
    ab_pred_2_inv = ab_pred_2 * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    ablation_results.append({
        "Variant": "w/o Attention",
        "MAE": mean_absolute_error(y_test_inv, ab_pred_2_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, ab_pred_2_inv)),
        "R2": r2_score(y_test_inv, ab_pred_2_inv),
        "MAPE": mape(y_test_inv, ab_pred_2_inv),
    })

    ab_no_both = LSTMWithAttention(len(ab_reduced_idx), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, use_attention=False)
    ab_no_both = train_model(ab_no_both,
                             X_train[:, :, ab_reduced_idx], y_train,
                             X_val[:, :, ab_reduced_idx], y_val)
    ab_pred_3, _ = predict_model(ab_no_both, X_test[:, :, ab_reduced_idx])
    ab_pred_3_inv = ab_pred_3 * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    ablation_results.append({
        "Variant": "w/o TSM & Attention",
        "MAE": mean_absolute_error(y_test_inv, ab_pred_3_inv),
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, ab_pred_3_inv)),
        "R2": r2_score(y_test_inv, ab_pred_3_inv),
        "MAPE": mape(y_test_inv, ab_pred_3_inv),
    })

    ablation_results.append({
        "Variant": "Full (SA-DTM-LSTM)",
        "MAE": results["SA-DTM-LSTM"]["MAE"],
        "RMSE": results["SA-DTM-LSTM"]["RMSE"],
        "R2": results["SA-DTM-LSTM"]["R2"],
        "MAPE": results["SA-DTM-LSTM"]["MAPE"],
    })

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(output_ablation, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Ablation Study Results")
    print("=" * 70)
    print(f"{'Variant':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE(%)':>8}")
    print("-" * 60)
    for _, row in ablation_df.iterrows():
        print(f"{row['Variant']:<25} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['R2']:>8.4f} {row['MAPE']:>8.2f}")
    print("=" * 70)

    print("\n[INFO] Diebold-Mariano Test (SA-DTM-LSTM vs others):")
    print(f"{'Comparison':<30} {'DM-stat':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 58)
    full_errors = y_test_inv - full_pred_inv
    for model_name in ["ARIMA", "GRU", "SA-LSTM", "DTM-LSTM"]:
        if model_name in all_predictions:
            other_pred = np.array(all_predictions[model_name])
            other_errors = y_test_inv - other_pred
            dm_stat, p_val = dm_test(full_errors, other_errors)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{'SA-DTM-LSTM vs ' + model_name:<30} {dm_stat:>10.4f} {p_val:>10.4f} {sig:>5}")

    print("\n[INFO] Computing feature importance (permutation-based)...")
    shap_importance = compute_shap_approximation(full_model, X_test, feature_names)

    shap_df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in sorted(shap_importance.items(), key=lambda x: -x[1])
    ])
    shap_df.to_csv(output_shap, index=False, encoding="utf-8-sig")

    print("\nFeature Importance (Permutation-based):")
    print(f"{'Feature':<15} {'Importance':>12}")
    print("-" * 28)
    for _, row in shap_df.iterrows():
        print(f"{row['feature']:<15} {row['importance']:>12.4f}")

    return results, ablation_results, shap_importance


if __name__ == "__main__":
    run_prediction()
