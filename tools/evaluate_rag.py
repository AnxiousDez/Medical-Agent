import sys
import time
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Ensure top-level src directory is on sys.path
SRC_DIR = (Path(__file__).resolve().parent.parent / 'src')
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embeddings.vector_store import MedicalVectorStore  # noqa: E402


def dcg(relevances: List[int]) -> float:
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    gains = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    ideal = sorted(gains, reverse=True)
    dcg_val = dcg(gains)
    idcg_val = dcg(ideal) if ideal else 0.0
    return (dcg_val / idcg_val) if idcg_val > 0 else 0.0


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    for idx, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / max(k, 1)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    ap_sum = 0.0
    hits = 0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            hits += 1
            ap_sum += hits / rank
    denom = min(len(relevant_ids), k)
    return ap_sum / denom if denom > 0 else 0.0


def diversity_at_k(retrieved_metadatas: List[Dict], k: int, field: str = 'doc_id') -> float:
    if not retrieved_metadatas:
        return 0.0
    topk = retrieved_metadatas[:k]
    unique_vals = {m.get(field, '') for m in topk if m.get(field, '')}
    return len(unique_vals) / max(k, 1)


def redundancy_at_k(retrieved_metadatas: List[Dict], k: int, field: str = 'doc_id') -> float:
    return 1.0 - diversity_at_k(retrieved_metadatas, k, field)


def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def ensure_dirs():
    Path('eval').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('figures').mkdir(parents=True, exist_ok=True)


def load_eval_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found at {path}. Copy eval/queries_template.csv to eval/queries.csv and fill it."
        )
    df = pd.read_csv(path)
    if 'query' not in df.columns or 'relevant_ids' not in df.columns:
        raise ValueError("CSV must contain columns: query, relevant_ids (semicolon-separated doc_ids)")
    return df


def run_retrieval_eval(eval_csv: Path, k_values: List[int] = [1, 3, 5, 10, 20]):
    ensure_dirs()

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)

    df = load_eval_data(eval_csv)

    store = MedicalVectorStore(persist_directory='./data/chroma_db')

    per_query_rows = []
    all_retrieved_lists: List[List[str]] = []
    all_relevant_sets: List[Set[str]] = []

    for _, row in df.iterrows():
        query = str(row['query']).strip()
        relevant_ids: Set[str] = set(str(row['relevant_ids']).split(';')) if pd.notna(row['relevant_ids']) else set()
        relevant_ids = {rid.strip() for rid in relevant_ids if rid.strip()}

        t0 = time.perf_counter()
        results = store.search_literature(query, n_results=max(k_values))
        dt_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids: List[str] = [m.get('doc_id', '') for m in results['metadatas']]
        retrieved_meta: List[Dict] = results['metadatas']

        metrics_row: Dict[str, float] = {
            'query': query,
            'relevant_count': len(relevant_ids),
            'latency_ms': dt_ms,
        }
        for k in k_values:
            # Standard retrieval metrics (doc-level relevance)
            p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
            r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics_row[f'P@{k}'] = p_at_k
            metrics_row[f'R@{k}'] = r_at_k
            metrics_row[f'NDCG@{k}'] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            metrics_row[f'MRR@{k}'] = mrr_at_k(retrieved_ids, relevant_ids, k)
            metrics_row[f'AP@{k}'] = average_precision_at_k(retrieved_ids, relevant_ids, k)
            metrics_row[f'Diversity@{k}'] = diversity_at_k(retrieved_meta, k)
            metrics_row[f'Redundancy@{k}'] = redundancy_at_k(retrieved_meta, k)
            metrics_row[f'F1@{k}'] = f1_at_k(retrieved_ids, relevant_ids, k)
            metrics_row[f'HitRate@{k}'] = hit_rate_at_k(retrieved_ids, relevant_ids, k)

            # Context-oriented metrics
            # ContextPrecision@k: proportion of retrieved chunks whose doc_id is relevant (same as P@k)
            metrics_row[f'ContextPrecision@{k}'] = p_at_k
            # ContextRecallUnique@k: fraction of unique relevant docs covered in top-k
            unique_hits = len(set(retrieved_ids[:k]).intersection(relevant_ids))
            metrics_row[f'ContextRecallUnique@{k}'] = (unique_hits / len(relevant_ids)) if len(relevant_ids) > 0 else 0.0
            # NoiseRate@k: fraction of retrieved chunks from non-relevant docs
            metrics_row[f'NoiseRate@{k}'] = 1.0 - p_at_k

        per_query_rows.append(metrics_row)
        all_retrieved_lists.append(retrieved_ids)
        all_relevant_sets.append(relevant_ids)

    per_query = pd.DataFrame(per_query_rows)
    per_query.to_csv('results/retrieval_per_query.csv', index=False)

    # Summary
    summary = per_query.drop(columns=['query']).mean(numeric_only=True).to_frame('mean')
    summary['std'] = per_query.drop(columns=['query']).std(numeric_only=True)

    # Add MAP@k rows (computed across queries, not mean of AP@k)
    map_rows = {}
    for k in [1, 3, 5, 10, 20]:
        maps = [average_precision_at_k(r, rel, k) for r, rel in zip(all_retrieved_lists, all_relevant_sets)]
        map_rows[f'MAP@{k}'] = {'mean': float(np.mean(maps)) if maps else 0.0, 'std': float(np.std(maps)) if maps else 0.0}
    map_df = pd.DataFrame.from_dict(map_rows, orient='index')

    summary = pd.concat([summary, map_df], axis=0)
    summary.to_csv('results/retrieval_summary.csv')

    print('Saved: results/retrieval_per_query.csv')
    print('Saved: results/retrieval_summary.csv')


def main():
    # Resolve eval CSV relative to project root (parent of tools/)
    eval_csv = (Path(__file__).resolve().parent.parent / 'eval' / 'queries.csv')
    run_retrieval_eval(eval_csv)


if __name__ == '__main__':
    main()


