import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    Path('figures').mkdir(parents=True, exist_ok=True)

    per_query_path = Path('results/generation_per_query.csv')
    summary_path = Path('results/generation_summary.csv')
    if not per_query_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run tools/evaluate_generation.py first to produce results CSVs.")

    per_query = pd.read_csv(per_query_path)
    summary = pd.read_csv(summary_path, index_col=0)

    # 1) Generation quality bar (mean ± std where available)
    gen_metrics = [m for m in [
        'Faithfulness', 'AnswerRelevancy', 'AnswerCorrectness_sem',
        'EM', 'F1', 'BLEU', 'ROUGE-L', 'METEOR'] if m in summary.index]
    if gen_metrics:
        df_bar = summary.loc[gen_metrics, ['mean']].reset_index().rename(columns={'index': 'metric'})
        plt.figure(figsize=(7,4))
        ax = sns.barplot(data=df_bar, x='metric', y='mean', color="#4C72B0")
        ax.set_ylim(0,1)
        ax.bar_label(ax.containers[0], fmt='%.2f', padding=2)
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout(); plt.savefig('figures/generation_metrics.svg'); plt.close()

    # 2) Context metrics bar
    ctx_metrics = [m for m in ['ContextPrecision', 'ContextRecall', 'ContextRelevancy', 'ContextEntitiesRecall'] if m in summary.index]
    if ctx_metrics:
        df_ctx = summary.loc[ctx_metrics, ['mean']].reset_index().rename(columns={'index': 'metric'})
        plt.figure(figsize=(7,4))
        ax = sns.barplot(data=df_ctx, x='metric', y='mean', color="#55A868")
        ax.set_ylim(0,1)
        ax.bar_label(ax.containers[0], fmt='%.2f', padding=2)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout(); plt.savefig('figures/context_metrics.svg'); plt.close()

    # 3) Latency scatter (retrieval vs generation)
    if 'latency_ms' in per_query.columns and 'gen_latency_ms' in per_query.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=per_query, x='latency_ms', y='gen_latency_ms')
        plt.xlabel('Retrieval latency (ms)'); plt.ylabel('Generation latency (ms)')
        plt.tight_layout(); plt.savefig('figures/latency_scatter.svg'); plt.close()

    print("Saved figures to ./figures/: generation_metrics.svg, context_metrics.svg, latency_scatter.svg")


if __name__ == '__main__':
    main()


