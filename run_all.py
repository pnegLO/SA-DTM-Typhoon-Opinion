import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("Step 1: Data Preprocessing", "preprocess", "preprocess"),
    ("Step 2: Sentiment Analysis", "sentiment_analysis", "analyze_sentiment"),
    ("Step 3: Dynamic Topic Modeling", "dtm_modeling", "run_dtm"),
    ("Step 4: Feature Fusion", "feature_fusion", "fuse_features"),
    ("Step 5: LSTM Prediction & Ablation", "lstm_prediction", "run_prediction"),
    ("Step 6: Visualization", "visualize", "generate_all_figures"),
]


def run_step(step_name, module_name, func_name):
    print("\n")
    print("=" * 70)
    print(f"  {step_name}")
    print("=" * 70)

    start = time.time()
    try:
        module = __import__(module_name)
        func = getattr(module, func_name)
        func()
        elapsed = time.time() - start
        print(f"\n[DONE] {step_name} completed in {elapsed:.1f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[FAIL] {step_name} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def print_summary(results):
    print("\n\n")
    print("=" * 70)
    print("  SA-DTM-LSTM PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    total_time = 0
    print(f"\n{'Step':<45} {'Status':>8} {'Time':>8}")
    print("-" * 65)
    for step_name, success, elapsed in results:
        status = "OK" if success else "FAIL"
        print(f"{step_name:<45} {status:>8} {elapsed:>7.1f}s")
        total_time += elapsed
    print("-" * 65)
    print(f"{'Total':<45} {'':>8} {total_time:>7.1f}s")

    print("\nOutput Files:")
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".csv"):
                size = os.path.getsize(os.path.join(data_dir, f))
                print(f"  data/{f} ({size/1024:.1f} KB)")

    fig_dir = "figures"
    if os.path.exists(fig_dir):
        fig_files = [f for f in os.listdir(fig_dir) if f.endswith((".png", ".pdf"))]
        print(f"\nFigures ({len(fig_files)} files):")
        for f in sorted(fig_files):
            size = os.path.getsize(os.path.join(fig_dir, f))
            print(f"  figures/{f} ({size/1024:.1f} KB)")

    if os.path.exists("data/model_metrics.csv"):
        import pandas as pd
        metrics = pd.read_csv("data/model_metrics.csv")
        print("\nModel Performance Summary:")
        print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE(%)':>8}")
        print("-" * 50)
        for _, row in metrics.iterrows():
            print(f"{row['Model']:<15} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
                  f"{row['R2']:>8.4f} {row['MAPE']:>8.2f}")

    if os.path.exists("data/ablation_results.csv"):
        import pandas as pd
        ab = pd.read_csv("data/ablation_results.csv")
        print("\nAblation Study Summary:")
        print(f"{'Variant':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
        print("-" * 50)
        for _, row in ab.iterrows():
            print(f"{row['Variant']:<25} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} {row['R2']:>8.4f}")

    all_ok = all(success for _, success, _ in results)
    print("\n" + "=" * 70)
    if all_ok:
        print("  ALL STEPS COMPLETED SUCCESSFULLY")
    else:
        failed = [name for name, success, _ in results if not success]
        print(f"  COMPLETED WITH ERRORS: {', '.join(failed)}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  SA-DTM-LSTM: Sentiment-Aware Dynamic Topic Model")
    print("  with LSTM for Public Opinion Prediction")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    pipeline_start = time.time()
    results = []

    for step_name, module_name, func_name in STEPS:
        success, elapsed = run_step(step_name, module_name, func_name)
        results.append((step_name, success, elapsed))

        if not success and module_name in ["preprocess", "sentiment_analysis", "dtm_modeling"]:
            print(f"\n[ABORT] Critical step failed: {step_name}")
            print("[INFO] Fix the error and re-run.")
            break

    print_summary(results)


if __name__ == "__main__":
    main()
