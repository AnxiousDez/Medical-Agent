import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    Path('figures').mkdir(parents=True, exist_ok=True)

    per_query_path = Path('results/retrieval_per_query.csv')
    summary_path = Path('results/retrieval_summary.csv')
    if not per_query_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run tools/evaluate_rag.py first to produce results CSVs.")

    per_query = pd.read_csv(per_query_path)
    summary = pd.read_csv(summary_path, index_col=0)

    # 1) Bar chart for mean metrics across selected ks
    metrics = [c for c in summary.index if c.startswith(('P@','R@','F1@','HitRate@','ContextPrecision@','ContextRecallUnique@','NoiseRate@'))]
    df_bar = summary.loc[metrics, ['mean']].reset_index().rename(columns={'index': 'metric'})
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df_bar, x='metric', y='mean', color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.bar_label(ax.containers[0], fmt='%.2f', padding=2)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.savefig('figures/retrieval_metrics.svg'); plt.close()

    # 2) Latency histogram
    if 'latency_ms' in per_query.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(per_query['latency_ms'], bins=20, kde=True)
        plt.xlabel('Latency (ms)'); plt.ylabel('Queries')
        plt.tight_layout(); plt.savefig('figures/latency_hist.svg'); plt.close()

    # 3) F1 vs k curve (derive k from summary rows F1@k)
    f1_rows = [r for r in summary.index if r.startswith('F1@')]
    if f1_rows:
        ks = [int(r.split('@')[1]) for r in f1_rows]
        f1_means = summary.loc[f1_rows, 'mean'].values
        order = sorted(range(len(ks)), key=lambda i: ks[i])
        ks_sorted = [ks[i] for i in order]
        f1_sorted = [f1_means[i] for i in order]
        plt.figure(figsize=(6, 4))
        plt.plot(ks_sorted, f1_sorted, marker='o', lw=2)
        plt.ylim(0, 1); plt.xlabel('k'); plt.ylabel('F1@k')
        plt.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig('figures/f1_vs_k.svg'); plt.close()

    # 4) Diversity vs k if present
    div_cols = [c for c in per_query.columns if c.startswith('Diversity@')]
    if div_cols:
        ks = [int(c.split('@')[1]) for c in div_cols]
        div_means = per_query[div_cols].mean().values
        order = sorted(range(len(ks)), key=lambda i: ks[i])
        ks_sorted = [ks[i] for i in order]
        divs_sorted = [div_means[i] for i in order]
        plt.figure(figsize=(6,4))
        plt.plot(ks_sorted, divs_sorted, marker='o', lw=2)
        plt.ylim(0,1); plt.xlabel('k'); plt.ylabel('Diversity@k')
        plt.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig('figures/diversity_vs_k.svg'); plt.close()

    print("Saved figures to ./figures/: retrieval_metrics.svg, latency_hist.svg, f1_vs_k.svg, diversity_vs_k.svg")


if __name__ == '__main__':
    main()


