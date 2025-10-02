import sys
import os
from pathlib import Path
import logging
import streamlit as st

# Ensure src on sys.path
SRC_DIR = str((Path(__file__).resolve().parent / 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.settings import settings  # noqa: E402


def _init_components():
    # Lazy import heavy modules after sys.path is set
    from embeddings.vector_store import MedicalVectorStore
    from data_processing.mimic_processor import MIMICProcessor
    from patient_data.patient_processor import PatientDataProcessor
    from rag.medical_rag import MedicalRAGSystem

    # Ensure required directories
    Path('logs').mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    vector_store = MedicalVectorStore(persist_directory=settings.CHROMA_DB_PATH)

    mimic_processor = MIMICProcessor(
        mimic_db_path=settings.MIMIC_DB_PATH,
        source=settings.MIMIC_SOURCE,
        csv_dir=settings.MIMIC_CSV_DIR,
    )

    patient_processor = PatientDataProcessor(mimic_processor, vector_store)

    rag_system = MedicalRAGSystem(
        vector_store=vector_store,
        patient_processor=patient_processor,
        openai_api_key=settings.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY', '')
    )

    return {
        'vector_store': vector_store,
        'mimic_processor': mimic_processor,
        'patient_processor': patient_processor,
        'rag_system': rag_system,
    }


@st.cache_resource(show_spinner=False)
def get_components():
    return _init_components()


def _render_sources(sources):
    if not sources:
        return
    st.subheader('Sources')
    for i, s in enumerate(sources, 1):
        if s.get('type') == 'literature':
            st.markdown(f"- 📄 {s.get('title','Medical Literature')} ({s.get('doc_type','')})")
        elif s.get('type') == 'patient':
            st.markdown(f"- 👤 Patient {s.get('patient_id','?')} — {s.get('summary','')} ")
        else:
            st.markdown(f"- 📊 {s}")


def create_streamlit_app():
    st.set_page_config(page_title='Medical RAG Assistant', layout='wide')
    st.title('🏥 Medical RAG Assistant')
    st.caption('Ask evidence-based questions using literature and patient data.')

    # Sidebar configuration
    with st.sidebar:
        st.header('Configuration')
        st.write('Environment')
        st.checkbox('Local-only mode (no external LLM calls)', value=settings.LOCAL_ONLY, key='local_only')
        model = st.text_input('LLM model', value=settings.LLM_MODEL)
        base_url = st.text_input('OpenAI Base URL (for Ollama set http://localhost:11434/v1)', value=settings.OPENAI_BASE_URL)
        st.markdown('---')
        st.write('Retrieval')
        n_results = st.slider('Top-k results', 1, 10, 5)
        st.markdown('---')
        st.write('Storage')
        st.code(f"Chroma: {settings.CHROMA_DB_PATH}")

    # Apply transient overrides
    os.environ['LOCAL_ONLY'] = 'true' if st.session_state.get('local_only') else 'false'
    if base_url:
        os.environ['OPENAI_BASE_URL'] = base_url
    if model:
        os.environ['LLM_MODEL'] = model

    components = get_components()
    rag = components['rag_system']

    # Main input
    with st.form('query_form'):
        query = st.text_area('Enter your medical question:', height=120, placeholder='e.g., What are first-line treatments for sepsis?')
        submitted = st.form_submit_button('Ask')

    if submitted and query.strip():
        with st.spinner('Retrieving and generating answer...'):
            try:
                response = rag.process_query(query)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader('Answer')
            st.write(response.answer)
        with col_b:
            st.subheader('Details')
            st.metric('Confidence', f"{response.confidence*100:.1f}%")
            st.caption(f"Query type: {response.query_type}")

        _render_sources(response.sources)

        # Optional: quick retrieval preview
        if st.checkbox('Show retrieved context preview'):
            ctx = rag.retrieve_context(query, response.query_type)
            st.write('Literature chunks:')
            for d in ctx.get('literature', [])[:5]:
                st.text(d[:500] + ('...' if len(d) > 500 else ''))
            if ctx.get('patients'):
                st.write('Patient summaries:')
                for d in ctx.get('patients', [])[:3]:
                    st.text(d[:400] + ('...' if len(d) > 400 else ''))


if __name__ == '__main__':
    create_streamlit_app()


