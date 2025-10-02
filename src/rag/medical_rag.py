# src/rag/medical_rag.py
from typing import Dict, List, Optional, Union
import openai
from dataclasses import dataclass
import re
import json

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]
    confidence: float
    query_type: str

class MedicalRAGSystem:
    def __init__(self, vector_store, patient_processor, openai_api_key: str):
        self.vector_store = vector_store
        self.patient_processor = patient_processor
        
        # Initialize OpenAI client
        from config.settings import settings
        if not settings.LOCAL_ONLY:
            if settings.OPENAI_BASE_URL:
                self.client = openai.OpenAI(api_key=openai_api_key, base_url=settings.OPENAI_BASE_URL)
            else:
                self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.client = None
        
        # Initialize NER for entity extraction
        try:
            from text_processing.medical_ner import AdvancedMedicalNER
            self.ner = AdvancedMedicalNER()
        except ImportError:
            print("Warning: Advanced NER not available, using basic processing")
            self.ner = None
        
        # Query classification patterns
        self.query_patterns = {
            'literature': [
                r'what does research say',
                r'latest studies',
                r'clinical trials',
                r'evidence for',
                r'treatment guidelines',
                r'research shows',
                r'studies indicate',
                r'literature review',
                r'systematic review'
            ],
            'patient_specific': [
                r'patient \d+',
                r'this patient',
                r'similar patients',
                r'patient with',
                r'find patients',
                r'patient cohort',
                r'patients who have',
                r'show me patient'
            ],
            'clinical_decision': [
                r'should I',
                r'what treatment',
                r'recommend',
                r'best approach',
                r'clinical decision',
                r'how to treat',
                r'management of',
                r'treatment plan'
            ],
            'drug_interaction': [
                r'interact with',
                r'drug interaction',
                r'can.*take.*with',
                r'contraindication',
                r'adverse reaction'
            ]
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type to route to appropriate retrieval"""
        query_lower = query.lower()
        
        # Check for drug interaction patterns first (most specific)
        for pattern in self.query_patterns['drug_interaction']:
            if re.search(pattern, query_lower):
                return 'drug_interaction'
        
        # Check for patient-specific patterns
        for pattern in self.query_patterns['patient_specific']:
            if re.search(pattern, query_lower):
                return 'patient_specific'
        
        # Check for literature patterns
        for pattern in self.query_patterns['literature']:
            if re.search(pattern, query_lower):
                return 'literature'
        
        # Check for clinical decision patterns
        for pattern in self.query_patterns['clinical_decision']:
            if re.search(pattern, query_lower):
                return 'clinical_decision'
        
        # Default to hybrid search
        return 'hybrid'
    
    def extract_patient_id(self, query: str) -> Optional[str]:
        """Extract patient ID from query if present"""
        patterns = [
            r'patient\s+(\d+)',
            r'patient\s+id\s+(\d+)',
            r'id\s*:\s*(\d+)',
            r'pt\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
        return None
    
    def extract_medications(self, query: str) -> List[str]:
        """Extract medication names from query"""
        common_medications = [
            'metformin', 'warfarin', 'aspirin', 'lisinopril', 'amlodipine',
            'atorvastatin', 'levothyroxine', 'omeprazole', 'albuterol',
            'hydrochlorothiazide', 'losartan', 'gabapentin', 'furosemide'
        ]
        
        found_meds = []
        query_lower = query.lower()
        
        for med in common_medications:
            if med in query_lower:
                found_meds.append(med)
        
        return found_meds
    
    def retrieve_context(self, query: str, query_type: str) -> Dict:
        """Retrieve relevant context based on query type"""
        context = {'literature': [], 'patients': [], 'metadata': []}
        
        if query_type == 'literature':
            # Search only literature
            lit_results = self.vector_store.search_literature(query, n_results=5)
            context['literature'] = lit_results['documents']
            context['metadata'].extend(lit_results['metadatas'])
            
        elif query_type == 'patient_specific':
            # Extract patient ID if present
            patient_id = self.extract_patient_id(query)
            
            if patient_id:
                try:
                    # Get specific patient data
                    patient_summary = self.patient_processor.mimic.create_patient_summary(patient_id)
                    context['patients'].append(str(patient_summary.get('summary_text', '')))
                    
                    # Find similar patients
                    similar_patients = self.patient_processor.find_similar_patients(
                        patient_id, {'diagnoses': True}, n_similar=3
                    )
                    context['patients'].extend([p['summary'] for p in similar_patients])
                except Exception as e:
                    print(f"Error retrieving patient data: {e}")
                    # Fallback to general patient search
                    patient_results = self.vector_store.search_patients(query, n_results=3)
                    context['patients'] = [str(d) for d in patient_results['documents']]
                    context['metadata'].extend(patient_results['metadatas'])
            else:
                # General patient search
                patient_results = self.vector_store.search_patients(query, n_results=5)
                context['patients'] = [str(d) for d in patient_results['documents']]
                context['metadata'].extend(patient_results['metadatas'])
                
        elif query_type == 'drug_interaction':
            # Search for drug interaction literature
            medications = self.extract_medications(query)
            if len(medications) >= 2:
                interaction_query = f"drug interaction {' and '.join(medications)}"
            else:
                interaction_query = f"drug interaction {query}"
            
            lit_results = self.vector_store.search_literature(interaction_query, n_results=5)
            context['literature'] = lit_results['documents']
            context['metadata'].extend(lit_results['metadatas'])
        
        elif query_type == 'clinical_decision':
            # Search both literature and patients
            lit_results = self.vector_store.search_literature(query, n_results=3)
            patient_results = self.vector_store.search_patients(query, n_results=3)
            
            context['literature'] = lit_results['documents']
            context['patients'] = patient_results['documents']
            context['metadata'].extend(lit_results['metadatas'] + patient_results['metadatas'])
        
        else:  # hybrid
            # Search all sources
            try:
                hybrid_results = self.vector_store.hybrid_search(query, n_results=3)
                if 'literature' in hybrid_results:
                    context['literature'] = hybrid_results['literature']['documents']
                    context['metadata'].extend(hybrid_results['literature']['metadatas'])
                if 'patients' in hybrid_results:
                    context['patients'] = hybrid_results['patients']['documents']
                    context['metadata'].extend(hybrid_results['patients']['metadatas'])
            except Exception as e:
                print(f"Error in hybrid search: {e}")
                # Fallback to literature only
                lit_results = self.vector_store.search_literature(query, n_results=5)
                context['literature'] = lit_results['documents']
                context['metadata'].extend(lit_results['metadatas'])
        
        return context
    
    def generate_response(self, query: str, context: Dict, query_type: str) -> str:
        """Generate response using retrieved context"""
        
        # Build context string
        context_parts = []
        
        if context['literature']:
            context_parts.append("MEDICAL LITERATURE:")
            for i, doc in enumerate(context['literature'][:3]):
                context_parts.append(f"Source {i+1}: {doc[:500]}...")
        
        if context['patients']:
            context_parts.append("\nPATIENT DATA:")
            for i, doc in enumerate(context['patients'][:3]):
                context_parts.append(f"Patient {i+1}: {doc[:400]}...")
        
        context_str = "\n".join(context_parts)
        
        # Create system prompt based on query type
        system_prompts = {
            'literature': """You are a medical AI assistant with access to medical literature. 
            Provide evidence-based answers citing the sources provided. Focus on:
            - Current research findings
            - Clinical evidence
            - Treatment guidelines
            Always cite your sources and indicate confidence levels.""",
            
            'patient_specific': """You are a medical AI assistant analyzing patient data.
            Provide insights based on the patient information provided. Focus on:
            - Patient-specific observations
            - Similar patient patterns
            - Relevant clinical correlations
            Always maintain patient privacy and provide objective analysis.""",
            
            'clinical_decision': """You are a clinical decision support AI assistant.
            Integrate both research evidence and patient data to provide:
            - Evidence-based recommendations
            - Patient-specific considerations
            - Risk-benefit analysis
            Always emphasize that final decisions should involve healthcare providers.""",
            
            'drug_interaction': """You are a pharmaceutical AI assistant specializing in drug interactions.
            Provide information about:
            - Known drug interactions
            - Mechanism of interaction
            - Clinical significance
            - Monitoring recommendations
            Always recommend consulting with a pharmacist or physician.""",
            
            'hybrid': """You are a comprehensive medical AI assistant.
            Use both literature and patient data to provide well-rounded answers.
            Clearly distinguish between research evidence and patient-specific insights."""
        }
        
        system_prompt = system_prompts.get(query_type, system_prompts['hybrid'])
        
        # Generate response using OpenAI
        from config.settings import settings
        if settings.LOCAL_ONLY or self.client is None:
            # Local-only summarization of retrieved context
            summary_parts = []
            if context['literature']:
                summary_parts.append("Evidence from literature:")
                for i, doc in enumerate(context['literature'][:3], 1):
                    summary_parts.append(f"{i}. {doc[:300]}...")
            if context['patients']:
                summary_parts.append("\nPatient data:")
                for i, doc in enumerate(context['patients'][:3], 1):
                    summary_parts.append(f"{i}. {doc[:250]}...")
            if not summary_parts:
                return "No relevant context was retrieved to answer the question locally."
            summary_parts.append("\nNote: Local-only mode is enabled; no LLM call was made.")
            return "\n".join(summary_parts)
        else:
            try:
                response = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
                        Context Information:
                        {context_str}
                        
                        Question: {query}
                        
                        Please provide a comprehensive answer based on the context provided.
                        If the context is limited, acknowledge this and provide general medical knowledge while being clear about the limitations.
                        """}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"I encountered an error generating the response: {str(e)}"
    
    def calculate_confidence(self, context: Dict, query: str) -> float:
        """Calculate confidence score based on retrieval quality"""
        base_confidence = 0.3
        
        # Boost confidence based on number of relevant sources
        total_sources = len(context['literature']) + len(context['patients'])
        if total_sources >= 5:
            base_confidence += 0.3
        elif total_sources >= 3:
            base_confidence += 0.2
        elif total_sources >= 1:
            base_confidence += 0.1
        
        # Check for medical entities in query (more specific = higher confidence)
        if self.ner:
            try:
                entities = self.ner.extract_comprehensive_entities(query)
                entity_count = sum(len(entities[key]) for key in entities)
                if entity_count >= 5:
                    base_confidence += 0.25
                elif entity_count >= 3:
                    base_confidence += 0.15
                elif entity_count >= 1:
                    base_confidence += 0.1
            except:
                # Basic entity counting fallback
                medical_terms = ['patient', 'treatment', 'medication', 'diagnosis', 'symptom']
                entity_count = sum(1 for term in medical_terms if term in query.lower())
                if entity_count >= 3:
                    base_confidence += 0.1
        
        # Boost confidence for specific query types
        if query.lower().count('?') == 1:  # Single clear question
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)  # Cap at 95%
    
    def process_query(self, query: str) -> RAGResponse:
        """Main method to process a query through the RAG pipeline"""
        
        # Step 1: Classify query
        query_type = self.classify_query(query)
        
        # Step 2: Retrieve relevant context
        context = self.retrieve_context(query, query_type)
        
        # Step 3: Generate response
        answer = self.generate_response(query, context, query_type)
        
        # Step 4: Calculate confidence
        confidence = self.calculate_confidence(context, query)
        
        # Step 5: Format sources
        sources = []
        for i, metadata in enumerate(context['metadata'][:5]):
            if isinstance(metadata, dict):
                if 'title' in metadata:
                    sources.append({
                        'type': 'literature',
                        'title': metadata['title'],
                        'doc_type': metadata.get('doc_type', 'unknown'),
                        'index': i
                    })
                elif 'patient_id' in metadata:
                    sources.append({
                        'type': 'patient',
                        'patient_id': metadata['patient_id'],
                        'summary': f"Patient data with {metadata.get('num_diagnoses', 0)} diagnoses",
                        'index': i
                    })
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query_type=query_type
        )