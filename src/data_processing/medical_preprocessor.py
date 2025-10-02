# src/data_processing/medical_preprocessor.py
import pandas as pd
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Load NLP model with graceful fallbacks
try:
    import spacy
    try:
        nlp = spacy.load("en_core_sci_sm")
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
except Exception:
    nlp = None

@dataclass
class MedicalDocument:
    """Structure for medical documents"""
    doc_id: str
    title: str
    content: str
    doc_type: str  # 'literature', 'patient_note', 'guideline'
    metadata: Dict
    
@dataclass
class PatientRecord:
    """Structure for patient records"""
    patient_id: str
    demographics: Dict
    diagnoses: List[str]
    medications: List[str]
    lab_results: Dict
    notes: List[str]
    metadata: Dict

class MedicalTextProcessor:
    def __init__(self):
        self.nlp = nlp
        
    def clean_medical_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize medical abbreviations
        abbreviations = {
            'pt': 'patient',
            'pts': 'patients',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'hx': 'history',
            'sx': 'symptoms'
        }
        
        for abbrev, full in abbreviations.items():
            text = re.sub(rf'\b{abbrev}\b', full, text, flags=re.IGNORECASE)
            
        return text.strip()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using spaCy"""
        if self.nlp is None:
            return {'diseases': [], 'medications': [], 'procedures': [], 'anatomy': []}
        doc = self.nlp(text)
        
        entities = {
            'diseases': [],
            'medications': [],
            'procedures': [],
            'anatomy': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'SYMPTOM']:
                entities['diseases'].append(ent.text)
            elif ent.label_ in ['CHEMICAL', 'DRUG']:
                entities['medications'].append(ent.text)
            elif ent.label_ in ['PROCEDURE']:
                entities['procedures'].append(ent.text)
            elif ent.label_ in ['ANATOMY']:
                entities['anatomy'].append(ent.text)
                
        return entities
    
    def chunk_medical_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split medical text into chunks with medical context preservation"""
        # Split by sentences first
        if self.nlp is None:
            # Simple fallback chunking by length if no NLP pipeline
            chunks = []
            text = re.sub(r'\s+', ' ', text).strip()
            i = 0
            while i < len(text):
                chunks.append(text[i:i+chunk_size])
                i += max(chunk_size - overlap, 1)
            return chunks
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                words = current_chunk.split()
                current_chunk = " ".join(words[-overlap:]) if len(words) > overlap else ""
            current_chunk += " " + sentence
            
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

# No side effects on import