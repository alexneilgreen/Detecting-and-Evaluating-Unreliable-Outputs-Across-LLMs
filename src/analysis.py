"""
Computes reliability metrics from query results CSV and generates
summary statistics and figures for cross-model comparison.

Metrics:
    - Response Accuracy
    - Self-Consistency
    - Semantic Variance (commented out)
    - Self-Reported Confidence

Usage:
    python analysis.py [--input PATH] [--output_dir PATH]
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances



# Load Results
def load_results(input_path):
    df = pd.read_csv(input_path)
    # Drop rows where API or parse errors occurred
    df = df[~df["Answer"].isin(["API_ERROR", "PARSE_ERROR"])]
    df = df[~df["Confidence"].isin(["API_ERROR", "PARSE_ERROR"])]
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    return df



# Metric Calculations

# Response Accuracy
def compute_accuracy(df):
    """
    Compares Answer column against Ground Truth Answer column.
    Returns a DataFrame with accuracy per model and dataset.
    """
    df = df.copy()
    df["Correct"] = df["Answer"].str.strip().str.lower() == df["Ground Truth Answer"].str.strip().str.lower()

    accuracy = (
        df.groupby(["AI Model", "Dataset"])["Correct"]
        .mean()
        .reset_index()
        .rename(columns={"Correct": "Accuracy"})
    )
    return accuracy

# Self Consistency
def compute_self_consistency(df):
    """
    For each model + dataset + question group, measures the frequency
    of the most common answer across trials.
    Returns a DataFrame with mean self-consistency per model and dataset.
    """
    def consistency_score(answers):
        if len(answers) == 1:
            return 1.0
        counts = answers.str.strip().str.lower().value_counts()
        return counts.iloc[0] / len(answers)

    consistency = (
        df.groupby(["AI Model", "Dataset", "Question Number"])["Answer"]
        .apply(consistency_score)
        .reset_index()
        .rename(columns={"Answer": "Consistency"})
    )

    mean_consistency = (
        consistency.groupby(["AI Model", "Dataset"])["Consistency"]
        .mean()
        .reset_index()
        .rename(columns={"Consistency": "Self-Consistency"})
    )
    return mean_consistency

# Semantic Variance
def compute_semantic_variance(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def group_variance(responses):
        if len(responses) < 2:
            return 0.0
        embeddings = model.encode(responses.tolist())
        distances = cosine_distances(embeddings)
        upper = distances[np.triu_indices_from(distances, k=1)]
        return float(np.mean(upper))

    variance = (
        df.groupby(["AI Model", "Dataset", "Question Number"])["Response"]
        .apply(group_variance)
        .reset_index()
        .rename(columns={"Response": "Semantic Variance"})
    )

    mean_variance = (
        variance.groupby(["AI Model", "Dataset"])["Semantic Variance"]
        .mean()
        .reset_index()
    )
    return mean_variance

# Self-Reported Confidence
def compute_confidence(df):
    """
    Averages the Confidence column per model and dataset.
    Returns a DataFrame with mean confidence per model and dataset.
    """
    confidence = (
        df.groupby(["AI Model", "Dataset"])["Confidence"]
        .mean()
        .reset_index()
        .rename(columns={"Confidence": "Mean Confidence"})
    )
    return confidence



# Tables and Plots

# Construct Metrics Table
def build_metrics_table(df):
    accuracy = compute_accuracy(df)
    consistency = compute_self_consistency(df)
    confidence = compute_confidence(df)
    variance = compute_semantic_variance(df)

    metrics = accuracy.merge(consistency, on=["AI Model", "Dataset"])
    metrics = metrics.merge(confidence, on=["AI Model", "Dataset"])
    metrics = metrics.merge(variance, on=["AI Model", "Dataset"])

    return metrics

# Generate Individual Plot
def plot_metric(metrics, metric_col, title, ylabel, output_path, higher_is_better=True):
    datasets = metrics["Dataset"].unique()
    models = metrics["AI Model"].unique()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = metrics[metrics["Dataset"] == dataset]
        subset = subset.set_index("AI Model").reindex(models)

        if subset[metric_col].eq("N/A").all():
            ax.text(0.5, 0.5, "Semantic Variance\nNot Yet Computed",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{title} - {dataset}")
            continue

        values = pd.to_numeric(subset[metric_col], errors="coerce")
        bars = ax.bar(x, values, width=0.6, color="steelblue", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(f"{title} - {dataset}")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1 if metric_col != "Mean Confidence" else 110)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[SAVED] {output_path}")

# Generate Plots for Question Specific Analysis
def plot_per_question_accuracy_confidence(df, output_dir):
    save_dir = os.path.join(output_dir, "per_question_analysis")
    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    df["Correct"] = df["Answer"].str.strip().str.lower() == df["Ground Truth Answer"].str.strip().str.lower()

    for dataset in df["Dataset"].unique():
        df_ds = df[df["Dataset"] == dataset]
        questions = sorted(df_ds["Question Number"].unique())
        x = np.arange(1, len(questions) + 1)

        for model in df_ds["AI Model"].unique():
            df_m = df_ds[df_ds["AI Model"] == model]
            grouped = df_m.groupby("Question Number")

            accuracy = grouped["Correct"].mean().reindex(questions).values
            confidence = grouped["Confidence"].mean().reindex(questions).values / 100

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.bar(x, accuracy, color="steelblue", alpha=0.6, label="Accuracy", width=0.4, align="center")
            ax.bar(x + 0.4, confidence, color="coral", alpha=0.6, label="Confidence (scaled 0-1)", width=0.4, align="center")

            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score")
            ax.set_xlabel("Question Number")
            ax.set_title(f"{model} - {dataset}: Per-Question Accuracy vs Confidence")
            ax.set_xticks(x + 0.2)
            ax.set_xticklabels(questions, fontsize=7, rotation=45)
            ax.legend(loc="upper right", fontsize=9)

            plt.tight_layout()
            filename = f"{model}_{dataset.lower().replace('-', '_')}.png"
            path = os.path.join(save_dir, filename)
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[SAVED] {path}")

# Generate Plots for All Metrics
def generate_figures(metrics, output_dir, df):
    plot_metric(
        metrics, "Accuracy", "Response Accuracy", "Accuracy (0-1)",
        os.path.join(output_dir, "accuracy.png")
    )
    plot_metric(
        metrics, "Self-Consistency", "Self-Consistency", "Consistency (0-1)",
        os.path.join(output_dir, "self_consistency.png")
    )
    plot_metric(
        metrics, "Semantic Variance", "Semantic Variance", "Mean Cosine Distance",
        os.path.join(output_dir, "semantic_variance.png"), higher_is_better=False
    )
    plot_metric(
        metrics, "Mean Confidence", "Mean Self-Reported Confidence", "Confidence (0-100)",
        os.path.join(output_dir, "confidence.png"), higher_is_better=True
    )
    plot_per_question_accuracy_confidence(df, output_dir)



# Main

def main():
    parser = argparse.ArgumentParser(
        description="Compute reliability metrics and generate figures from query results."
    )
    parser.add_argument(
        "--input", type=str, default="results/results.csv",
        help="Path to the results CSV produced by query_models.py (default: results/results.csv)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/analysis",
        help="Directory to save metrics CSV and figures (default: results/analysis)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[LOADING] Results from {args.input}...")
    df = load_results(args.input)
    print(f"[LOADING] {len(df)} valid rows.")
    metrics = build_metrics_table(df)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    generate_figures(metrics, args.output_dir, df)
    print("\n[SAVED] Figures and Analysis")

if __name__ == "__main__":
    main()