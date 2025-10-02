import sys
from pathlib import Path

# Ensure top-level src directory is on sys.path
SRC_DIR = (Path(__file__).resolve().parent.parent / 'src')
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import List

from data_processing.medical_preprocessor import MedicalDocument
from embeddings.vector_store import MedicalVectorStore

LIT_DIR = Path('data/literature')


def collect_documents() -> List[MedicalDocument]:
    docs: List[MedicalDocument] = []
    if not LIT_DIR.exists():
        return docs
    for path in LIT_DIR.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.txt', '.md'}:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        name = path.stem.replace(' ', '_').lower()
        doc_type = 'clinical_guideline' if 'guideline' in name or 'guidelines' in name else 'research_summary'
        docs.append(MedicalDocument(
            doc_id=name,
            title=path.stem,
            content=text,
            doc_type=doc_type,
            metadata={'source_file': str(path)}
        ))
    return docs


def main():
    vector_store = MedicalVectorStore(persist_directory='./data/chroma_db')
    docs = collect_documents()
    if not docs:
        print('No .txt/.md literature files found in data/literature')
        return
    vector_store.add_literature_documents(docs)
    print(f'✓ Ingested {len(docs)} literature documents into ChromaDB')


if __name__ == '__main__':
    main()


