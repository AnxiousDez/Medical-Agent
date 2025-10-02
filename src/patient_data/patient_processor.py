# src/patient_data/patient_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from data_processing.mimic_processor import MIMICProcessor
from text_processing.medical_ner import AdvancedMedicalNER
from data_processing.medical_preprocessor import PatientRecord
import json

class PatientDataProcessor:
    def __init__(self, mimic_processor: MIMICProcessor, vector_store):
        self.mimic = mimic_processor
        self.vector_store = vector_store
        self.ner = AdvancedMedicalNER()
        
    def create_patient_embedding_data(self, patient_ids: List[str]) -> List[PatientRecord]:
        """Create comprehensive patient records for embedding"""
        patient_records = []
        
        for patient_id in patient_ids:
            try:
                # Get comprehensive patient data
                summary = self.mimic.create_patient_summary(patient_id)
                
                # Process clinical notes with NER
                processed_notes = []
                if summary['notes']:
                    for note in summary['notes']:
                        entities = self.ner.extract_comprehensive_entities(note['text'])
                        processed_note = {
                            'original_text': note['text'][:500],  # Truncate for storage
                            'entities': entities,
                            'category': note['category'],
                            'charttime': note['charttime']
                        }
                        processed_notes.append(processed_note)
                
                # Create patient record
                patient_record = PatientRecord(
                    patient_id=patient_id,
                    demographics=summary['demographics'],
                    diagnoses=[dx['diagnosis_description'] for dx in summary['diagnoses']],
                    medications=[med['drug'] for med in summary['medications']],
                    lab_results={},  # Would need lab processing
                    notes=[note['original_text'] for note in processed_notes],
                    metadata={
                        'num_admissions': len(summary['diagnoses']),
                        'processed_notes': processed_notes,
                        'last_updated': datetime.now().isoformat()
                    }
                )
                
                patient_records.append(patient_record)
                
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue
        
        return patient_records
    
    def find_similar_patients(self, target_patient_id: str, 
                             similarity_criteria: Dict, 
                             n_similar: int = 5) -> List[Dict]:
        """Find patients similar to target patient based on criteria"""
        
        # Get target patient data
        target_summary = self.mimic.create_patient_summary(target_patient_id)
        
        # Define similarity search query
        if similarity_criteria.get('diagnoses'):
            # Search by diagnosis similarity
            diagnosis_text = ', '.join([
                (dx.get('diagnosis_description') if isinstance(dx, dict) else str(dx))
                for dx in target_summary.get('diagnoses', [])[:5]
            ])
            search_query = f"Patient with diagnoses: {diagnosis_text}"
            
        elif similarity_criteria.get('medications'):
            # Search by medication similarity
            med_text = ', '.join([
                (med.get('drug') if isinstance(med, dict) else str(med))
                for med in target_summary.get('medications', [])[:5]
            ])
            search_query = f"Patient taking medications: {med_text}"
            
        else:
            # General similarity search
            search_query = target_summary['summary_text']
        
        # Search in vector store
        similar_results = self.vector_store.search_patients(
            query=search_query,
            n_results=n_similar + 1  # +1 to exclude self
        )
        
        # Filter out the target patient and format results
        similar_patients = []
        for i, patient_id in enumerate(similar_results['metadatas']):
            if patient_id['patient_id'] != target_patient_id:
                similar_patients.append({
                    'patient_id': patient_id['patient_id'],
                    'similarity_score': 1 - similar_results['distances'][i],  # Convert distance to similarity
                    'summary': similar_results['documents'][i][:200] + "...",
                    'metadata': patient_id
                })
        
        return similar_patients[:n_similar]
    
    def generate_patient_cohort(self, criteria: Dict) -> List[str]:
        """Generate a cohort of patients based on specific criteria"""
        
        # Example criteria:
        # {
        #     'age_range': (18, 65),
        #     'gender': 'M',
        #     'diagnoses': ['diabetes', 'hypertension'],
        #     'medications': ['metformin'],
        #     'admission_count_min': 2
        # }
        
        where_filters = {}
        
        if criteria.get('age_range'):
            min_age, max_age = criteria['age_range']
            # This would require adding age to metadata during ingestion
            
        if criteria.get('diagnoses'):
            # Search for patients with specific diagnoses
            diagnosis_query = ' OR '.join(criteria['diagnoses'])
            results = vector_store.search_patients(
                query=f"diagnosed with {diagnosis_query}",
                n_results=1000
            )
            return [meta['patient_id'] for meta in results['metadatas']]
        
        # For demonstration, return a sample
        return ['12345', '12346', '12347']  # Would be actual patient IDs

# No side effects on import