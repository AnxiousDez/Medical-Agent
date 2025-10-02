# src/embeddings/vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from data_processing.medical_preprocessor import MedicalDocument, PatientRecord, MedicalTextProcessor
import uuid
import torch

class MedicalVectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Select device (GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Use medical domain-specific embedding model (on selected device)
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=device
        )
        try:
            # Explicitly move to device (defensive; SentenceTransformer handles device internally)
            self.embedding_model.to(device)
        except Exception:
            pass

        # Optional: print device for visibility
        print(f"[VectorStore] Embedding device: {device}")
        # Alternative: Use BioBERT for better medical understanding
        # self.embedding_model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
        
        # Create collections for different data types
        self.literature_collection = self.client.get_or_create_collection(
            name="medical_literature",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.patient_collection = self.client.get_or_create_collection(
            name="patient_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text processor for chunking
        self.processor = MedicalTextProcessor()
        
    def add_literature_documents(self, documents: List[MedicalDocument]):
        """Add literature documents to vector store"""
        texts = []
        metadatas = []
        ids = []
        
        for doc in documents:
            # Chunk long documents
            chunks = self.processor.chunk_medical_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.doc_id}_chunk_{i}"
                texts.append(chunk)
                metadatas.append({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'doc_type': doc.doc_type,
                    'chunk_index': i,
                    **doc.metadata
                })
                ids.append(chunk_id)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False).tolist()
        
        # Add to collection
        self.literature_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
    def add_patient_records(self, patient_records: List[PatientRecord]):
        """Add patient records to vector store"""
        texts = []
        metadatas = []
        ids = []
        
        for record in patient_records:
            # Create searchable text from structured data
            patient_text = self._create_patient_text(record)
            
            texts.append(patient_text)
            metadatas.append({
                'patient_id': record.patient_id,
                'has_demographics': bool(record.demographics),
                'num_diagnoses': len(record.diagnoses),
                'num_medications': len(record.medications),
                **record.metadata
            })
            ids.append(record.patient_id)
        
        embeddings = self.embedding_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False).tolist()
        
        self.patient_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def _create_patient_text(self, record: PatientRecord) -> str:
        """Convert structured patient data to searchable text"""
        text_parts = []
        
        # Demographics
        if record.demographics:
            demo = record.demographics
            text_parts.append(f"Patient demographics: {demo.get('gender', '')} age {demo.get('age', '')}")
        
        # Diagnoses
        if record.diagnoses:
            text_parts.append(f"Diagnosed with: {', '.join(record.diagnoses[:10])}")
        
        # Medications
        if record.medications:
            text_parts.append(f"Current medications: {', '.join(record.medications[:15])}")
        
        # Clinical notes (summarized)
        if record.notes:
            notes_text = ' '.join(record.notes[:3])  # Limit to prevent huge embeddings
            if len(notes_text) > 1000:
                notes_text = notes_text[:1000] + "..."
            text_parts.append(f"Clinical notes: {notes_text}")
            
        return ". ".join(text_parts)
    
    def search_literature(self, query: str, n_results: int = 5, filters: Dict = None) -> Dict:
        """Search medical literature"""
        query_embedding = self.embedding_model.encode([query], batch_size=1, convert_to_numpy=True, show_progress_bar=False).tolist()
        
        where_clause = {}
        if filters:
            where_clause.update(filters)
        
        results = self.literature_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def search_patients(self, query: str, n_results: int = 5, filters: Dict = None) -> Dict:
        """Search patient records"""
        query_embedding = self.embedding_model.encode([query], batch_size=1, convert_to_numpy=True, show_progress_bar=False).tolist()
        
        where_clause = {}
        if filters:
            where_clause.update(filters)
        
        results = self.patient_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def hybrid_search(self, query: str, search_literature: bool = True, 
                     search_patients: bool = True, n_results: int = 3) -> Dict:
        """Search both literature and patient data"""
        results = {}
        
        if search_literature:
            results['literature'] = self.search_literature(query, n_results)
            
        if search_patients:
            results['patients'] = self.search_patients(query, n_results)
            
        return results

# No side effects on import