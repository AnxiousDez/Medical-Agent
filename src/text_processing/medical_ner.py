# src/text_processing/medical_ner.py
from typing import Dict, List
import re

class AdvancedMedicalNER:
    def __init__(self):
        # Lazy import spaCy and try SciSpaCy; fall back to en_core_web_sm if unavailable
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                try:
                    # Optional SciSpaCy linker
                    self.nlp.add_pipe(
                        "scispacy_linker", 
                        config={"resolve_abbreviations": True, "linker_name": "umls"}
                    )
                except Exception:
                    pass
            except Exception:
                # Fallback to general English model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # Final fallback: create blank English pipeline with sentencizer
                    self.nlp = spacy.blank("en")
                    if "sentencizer" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("sentencizer")
        except Exception:
            # If spaCy isn't available at all, create a minimal stub
            self.nlp = None
        
        # Define medical entity patterns
        self.medication_patterns = [
            r'\b\d+\s?mg\b',  # Dosages
            r'\b\d+\s?mcg\b',
            r'\btablet[s]?\b',
            r'\bcapsule[s]?\b',
            r'\binjection[s]?\b'
        ]
        
        self.vital_patterns = [
            r'BP\s?\d+/\d+',  # Blood pressure
            r'HR\s?\d+',      # Heart rate
            r'temp\s?\d+\.?\d*', # Temperature
            r'\d+\s?bpm',     # Beats per minute
        ]
    
    def extract_comprehensive_entities(self, text: str) -> Dict:
        """Extract all medical entities with confidence scores"""
        doc = self.nlp(text)
        
        entities = {
            'diseases': [],
            'medications': [],
            'procedures': [],
            'anatomy': [],
            'lab_values': [],
            'vitals': [],
            'dosages': []
        }
        
        # Extract named entities
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in getattr(doc, 'ents', []):
                entity_info = {
                    'text': ent.text,
                    'label': getattr(ent, 'label_', ''),
                    'start': getattr(ent, 'start_char', 0),
                    'end': getattr(ent, 'end_char', 0),
                    'confidence': getattr(ent, 'score', 0.5)
                }
                # Optional UMLS linking
                try:
                    kb_ents = getattr(ent._, 'kb_ents', [])
                    if kb_ents:
                        entity_info['umls_id'] = kb_ents[0][0]
                        entity_info['umls_score'] = kb_ents[0][1]
                except Exception:
                    pass
                # Categorize entities conservatively
                label = entity_info['label'].upper()
                if label in ['DISEASE', 'SYMPTOM', 'SIGN']:
                    entities['diseases'].append(entity_info)
                elif label in ['CHEMICAL', 'DRUG']:
                    entities['medications'].append(entity_info)
                elif label in ['PROCEDURE', 'TREATMENT']:
                    entities['procedures'].append(entity_info)
                elif label in ['ANATOMY', 'ORGAN']:
                    entities['anatomy'].append(entity_info)
        
        # Extract medication dosages with regex
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['dosages'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'dosage'
                })
        
        # Extract vital signs
        for pattern in self.vital_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['vitals'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'vital_sign'
                })
        
        return entities
    
    def normalize_medical_terms(self, text: str) -> str:
        """Normalize medical abbreviations and terms"""
        # Medical abbreviation dictionary
        abbreviations = {
            # Common medical abbreviations
            r'\bpt\.?\b': 'patient',
            r'\bpts\.?\b': 'patients',
            r'\bdx\b': 'diagnosis',
            r'\btx\b': 'treatment',
            r'\bhx\b': 'history',
            r'\bsx\b': 'symptoms',
            r'\bRx\b': 'prescription',
            r'\bBP\b': 'blood pressure',
            r'\bHR\b': 'heart rate',
            r'\bRR\b': 'respiratory rate',
            r'\bO2\b': 'oxygen',
            r'\bCO2\b': 'carbon dioxide',
            r'\bEKG\b': 'electrocardiogram',
            r'\bECG\b': 'electrocardiogram',
            r'\bCT\b': 'computed tomography',
            r'\bMRI\b': 'magnetic resonance imaging',
            r'\bICU\b': 'intensive care unit',
            r'\bER\b': 'emergency room',
            r'\bOR\b': 'operating room',
            r'\bNPO\b': 'nothing by mouth',
            r'\bprn\b': 'as needed',
            r'\bbid\b': 'twice daily',
            r'\btid\b': 'three times daily',
            r'\bqid\b': 'four times daily',
        }
        
        normalized_text = text
        for abbrev, expansion in abbreviations.items():
            normalized_text = re.sub(abbrev, expansion, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def extract_medication_regimen(self, text: str) -> List[Dict]:
        """Extract structured medication information"""
        medications = []
        
        # Pattern for medication with dosage and frequency
        med_pattern = r'(\w+(?:\s+\w+))\s(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?))\s*(?:(?:every|q)\s*(\d+)\s*(?:hours?|hrs?|h)|(\w+(?:\s+\w+)*frequency))'
        
        matches = re.finditer(med_pattern, text, re.IGNORECASE)
        for match in matches:
            medication = {
                'name': match.group(1).strip(),
                'dose': match.group(2).strip(),
                'frequency': match.group(3) or match.group(4),
                'full_text': match.group(0)
            }
            medications.append(medication)
        
        return medications

# No side effects on import