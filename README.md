## Medical-Agent: Retrieval-Augmented Generation for Clinical Data and Literature

Medical-Agent is a Retrieval-Augmented Generation (RAG) toolkit for clinical questions. It supports:

- Ingesting medical literature and synthetic EMR data (Synthea, MIMIC-like CSVs)
- Building and querying a local vector store
- Running an interactive Streamlit UI for question answering
- Evaluating retrieval and generation quality with provided tools


### Quickstart

Prerequisites:
- Python 3.9–3.11
- Git
- (Optional) `git-lfs` if you intend to version large files

Install dependencies (editable mode):

```powershell
cd "C:\Users\Aksha\Desktop\Medical rag"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

Basic smoke test:

```powershell
python -c "import src; print('OK')"
```


### Project layout

```
Medical rag/
├─ src/                          # Package code
│  ├─ data_processing/           # Literature + MIMIC/Synthea processors
│  ├─ embeddings/                # Vector store utilities
│  ├─ patient_data/              # Patient processors/utilities
│  ├─ query_preprocessing/       # Query routing and preprocessing
│  ├─ rag/                       # RAG orchestration
│  └─ text_processing/           # NER and text utilities
├─ tools/                        # CLIs for ingestion, evaluation, plotting
├─ data/                         # Local datasets & vector DBs (gitignored)
│  ├─ literature/                # Markdown articles
│  ├─ mimic/                     # CSVs and SQLite (local only)
│  └─ chroma_db/                 # ChromaDB files
├─ figures/                      # Generated plots (gitignored)
├─ results/                      # Evaluation output (gitignored)
├─ streamlit_app.py              # Interactive RAG UI
├─ main.py                       # Script entry (batch runs/experiments)
├─ setup.py                      # Package metadata & deps
├─ test_data.py                  # Sample tests/utilities
├─ README.md
└─ .gitignore
```


### Configure settings

Check `config/settings.py` for environment variables, model choices, and paths. Common options include:

- Embedding model and chunking parameters
- Vector store (Chroma) location under `data/chroma_db/`
- Retrieval `k`, reranking, and filtering settings


### Ingest data

1) Literature (Markdown files already included under `data/literature/`):

```powershell
python tools/ingest_literature.py \
  --input-dir data/literature \
  --db-dir data/chroma_db
```

2) Synthea / MIMIC-like data (optional):

- Synthea JSON is present under `synthea-master-branch-latest/output/`.
- MIMIC-like CSVs live under `data/mimic/`.

There are helper scripts to transform/augment:

```powershell
python tools/synthea_to_mimic.py --input synthea-master-branch-latest/output --output data/mimic
python tools/augment_mimic_csv.py --input data/mimic --output data/mimic
```

Note: Large datasets, CSVs, and databases are intentionally gitignored.


### Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

This launches an interactive UI where you can enter clinical queries. The app queries the local vector store built from your data and displays retrieved contexts and generated answers.


### CLI usage (batch/experiments)

Run the main script for end-to-end RAG or batch experiments:

```powershell
python main.py --help
```

Typical workflow:

```powershell
# Preprocess/ingest if needed (see above)
# Then run retrieval/generation experiments
python tools/evaluate_rag.py --queries eval/queries.csv --db-dir data/chroma_db --results-dir results

# Plot evaluation figures
python tools/generate_figures.py
```

Generated outputs will be placed under `results/` and `figures/`.


### Development

- Install in editable mode: `pip install -e .`
- Run specific modules directly, e.g.:

```powershell
python -m src.embeddings.vector_store
```

- Suggested style: black/ruff (optional). Add them to your environment if you prefer linting/formatting.


### Data and storage

- Vector store: ChromaDB at `data/chroma_db/` (local only)
- Large files: CSVs/SQLite under `data/` are excluded by `.gitignore`
- If you need to version large artifacts, consider `git lfs` and update `.gitattributes`


### Troubleshooting

- Push rejected with “fetch first”: the remote already has commits. Use:

```powershell
git fetch origin
git pull --rebase origin main --allow-unrelated-histories
git push -u origin main
```

- File too large for GitHub (>100MB): remove from history and ignore:

```powershell
git rm --cached path\to\large_file
echo "path/to/large_file" >> .gitignore
git commit -m "Stop tracking large file"
git push
```


### License

Add your preferred license (e.g., MIT) at the repository root as `LICENSE`.


### Acknowledgements

- Synthea synthetic EMR data generator
- ChromaDB for local vector storage

