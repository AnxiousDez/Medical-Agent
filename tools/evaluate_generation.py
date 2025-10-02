import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional metrics libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer  # type: ignore
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score  # type: ignore
    METEOR_AVAILABLE = True
except Exception:
    METEOR_AVAILABLE = False


# Ensure project root and top-level src directory are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = (ROOT_DIR / 'src')
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embeddings.vector_store import MedicalVectorStore  # noqa: E402
from patient_data.patient_processor import PatientDataProcessor  # noqa: E402
from data_processing.mimic_processor import MIMICProcessor  # noqa: E402
from rag.medical_rag import MedicalRAGSystem  # noqa: E402
from config.settings import settings  # noqa: E402


def ensure_dirs():
    Path('eval').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('figures').mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> List[str]:
    return [t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split() if t]


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def f1_token_overlap(pred: str, ref: str) -> float:
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = {}
    for t in ref_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / max(len(pred_toks), 1)
    recall = overlap / max(len(ref_toks), 1)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def bleu_score(pred: str, ref: str) -> Optional[float]:
    if not NLTK_AVAILABLE:
        return None
    smoothie = SmoothingFunction().method3
    return float(sentence_bleu([tokenize(ref)], tokenize(pred), smoothing_function=smoothie))


def rouge_l_score(pred: str, ref: str) -> Optional[float]:
    if not ROUGE_AVAILABLE:
        return None
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return float(scores['rougeL'].fmeasure)


def meteor(pred: str, ref: str) -> Optional[float]:
    if not METEOR_AVAILABLE:
        return None
    return float(meteor_score([ref], pred))


def average_max_cosine(query: str, docs: List[str], encoder) -> float:
    if not docs:
        return 0.0
    try:
        q = encoder.encode([query])[0]
        D = encoder.encode(docs)
        qn = q / (np.linalg.norm(q) + 1e-9)
        Dn = np.array([d / (np.linalg.norm(d) + 1e-9) for d in D])
        sims = Dn @ qn
        return float(np.max(sims))
    except Exception:
        return 0.0


def sentence_level_faithfulness(answer: str, ctx_docs: List[str], encoder) -> float:
    if not answer.strip() or not ctx_docs:
        return 0.0
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if not sentences:
        sentences = [answer.strip()]
    try:
        ans_emb = encoder.encode(sentences)
        ctx_emb = encoder.encode(ctx_docs)
        ctx_norm = np.array([v / (np.linalg.norm(v) + 1e-9) for v in ctx_emb])
        faith_scores: List[float] = []
        for v in ans_emb:
            v = v / (np.linalg.norm(v) + 1e-9)
            sims = ctx_norm @ v
            faith_scores.append(float(np.max(sims)))
        return float(np.mean(faith_scores)) if faith_scores else 0.0
    except Exception:
        return 0.0


def compute_context_precision_recall(retrieved_doc_ids: List[str], relevant_ids: List[str], k: int) -> Tuple[float, float]:
    topk = retrieved_doc_ids[:k]
    rel = set([r.strip() for r in relevant_ids if r and str(r).strip()])
    if not topk:
        return 0.0, 0.0
    hits = sum(1 for rid in topk if rid in rel)
    precision = hits / max(len(topk), 1)
    recall = hits / max(len(rel), 1) if rel else 0.0
    return precision, recall


def compute_entities_recall(context_docs: List[str], key_entities: List[str]) -> float:
    if not key_entities:
        return 0.0
    hay = ' \n '.join(context_docs).lower()
    entities = [e.strip().lower() for e in key_entities if e and e.strip()]
    if not entities:
        return 0.0
    found = sum(1 for e in entities if e in hay)
    return found / max(len(entities), 1)


def load_eval_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found at {path}. Expected columns: query, relevant_ids; optional: reference_answer, key_entities"
        )
    df = pd.read_csv(path)
    if 'query' not in df.columns or 'relevant_ids' not in df.columns:
        raise ValueError("CSV must contain columns: query, relevant_ids")
    # Normalize optional columns
    if 'reference_answer' not in df.columns:
        df['reference_answer'] = ''
    if 'key_entities' not in df.columns:
        df['key_entities'] = ''
    return df


def build_components():
    vector_store = MedicalVectorStore(persist_directory=settings.CHROMA_DB_PATH)
    mimic_processor = MIMICProcessor(
        mimic_db_path=settings.MIMIC_DB_PATH,
        source=settings.MIMIC_SOURCE,
        csv_dir=settings.MIMIC_CSV_DIR,
    )
    patient_processor = PatientDataProcessor(mimic_processor, vector_store)
    rag = MedicalRAGSystem(
        vector_store=vector_store,
        patient_processor=patient_processor,
        openai_api_key=settings.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY', ''),
    )
    return vector_store, rag


def run_generation_eval(eval_csv: Path, k_eval: int = 5):
    ensure_dirs()

    df = load_eval_data(eval_csv)

    vector_store, rag = build_components()
    encoder = vector_store.embedding_model

    rows: List[Dict] = []

    for _, row in df.iterrows():
        query = str(row['query']).strip()
        relevant_ids = [r.strip() for r in str(row['relevant_ids']).split(';') if r and str(r).strip()]
        ref_answer = str(row.get('reference_answer', '') or '').strip()
        key_entities = [e.strip() for e in str(row.get('key_entities', '') or '').split(';') if e and e.strip()]

        # Retrieval for evaluation
        t0 = time.perf_counter()
        ret = vector_store.search_literature(query, n_results=max(k_eval, 5))
        latency_ms = (time.perf_counter() - t0) * 1000
        retrieved_ids = [m.get('doc_id', '') for m in ret['metadatas']]
        retrieved_docs = ret['documents']

        # RAG generation
        t1 = time.perf_counter()
        resp = rag.process_query(query)
        gen_latency_ms = (time.perf_counter() - t1) * 1000
        answer = resp.answer or ''

        # Context metrics
        ctx_prec, ctx_rec = compute_context_precision_recall(retrieved_ids, relevant_ids, k_eval)
        ctx_relevancy = average_max_cosine(query, retrieved_docs[:k_eval], encoder)
        ctx_entities_recall = compute_entities_recall(retrieved_docs[:k_eval], key_entities)

        # Generation metrics (referential)
        em = f1 = bleu = rougeL = meteor_v = None
        if ref_answer:
            em = exact_match(answer, ref_answer)
            f1 = f1_token_overlap(answer, ref_answer)
            bleu = bleu_score(answer, ref_answer)
            rougeL = rouge_l_score(answer, ref_answer)
            meteor_v = meteor(answer, ref_answer)

        # Semantic metrics (no ground truth needed)
        faithfulness = sentence_level_faithfulness(answer, retrieved_docs[:k_eval], encoder)
        answer_relevancy = average_max_cosine(answer, [query], encoder)
        correctness_semantic = average_max_cosine(answer, [ref_answer], encoder) if ref_answer else 0.0

        rows.append({
            'query': query,
            'latency_ms': latency_ms,
            'gen_latency_ms': gen_latency_ms,
            'ContextPrecision': ctx_prec,
            'ContextRecall': ctx_rec,
            'ContextRelevancy': ctx_relevancy,
            'ContextEntitiesRecall': ctx_entities_recall,
            'Faithfulness': faithfulness,
            'AnswerRelevancy': answer_relevancy,
            'AnswerCorrectness_sem': correctness_semantic,
            'EM': em if em is not None else np.nan,
            'F1': f1 if f1 is not None else np.nan,
            'BLEU': bleu if bleu is not None else np.nan,
            'ROUGE-L': rougeL if rougeL is not None else np.nan,
            'METEOR': meteor_v if meteor_v is not None else np.nan,
        })

    per_query = pd.DataFrame(rows)
    per_query.to_csv('results/generation_per_query.csv', index=False)

    # Summary (mean/std)
    numeric = per_query.select_dtypes(include=[float, int])
    summary = numeric.mean().to_frame('mean')
    summary['std'] = numeric.std()
    summary.to_csv('results/generation_summary.csv')

    print('Saved: results/generation_per_query.csv')
    print('Saved: results/generation_summary.csv')


def main():
    # Resolve eval CSV relative to project root (parent of tools/)
    eval_csv = (Path(__file__).resolve().parent.parent / 'eval' / 'queries.csv')
    run_generation_eval(eval_csv)


if __name__ == '__main__':
    main()


