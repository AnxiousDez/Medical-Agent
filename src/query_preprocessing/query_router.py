# src/query_processing/query_router.py
import re
from typing import Dict, List, Tuple, Optional
from rag.medical_rag import RAGResponse
from enum import Enum

class QueryIntent(Enum):
    LITERATURE_SEARCH = "literature_search"
    PATIENT_LOOKUP = "patient_lookup"
    PATIENT_SIMILARITY = "patient_similarity"
    CLINICAL_DECISION = "clinical_decision"
    DRUG_INTERACTION = "drug_interaction"
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    COHORT_ANALYSIS = "cohort_analysis"

class QueryRouter:
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.LITERATURE_SEARCH: [
                r'research on|studies about|literature review|evidence for|papers on',
                r'what does research say|latest findings|clinical evidence',
                r'systematic review|meta.analysis|randomized trial'
            ],
            QueryIntent.PATIENT_LOOKUP: [
                r'patient (\d+)|patient id (\d+)|show me patient',
                r'patient information|patient data|patient summary'
            ],
            QueryIntent.PATIENT_SIMILARITY: [
                r'similar patients|patients like|find patients with',
                r'comparable cases|matching patients|similar profiles'
            ],
            QueryIntent.CLINICAL_DECISION: [
                r'should i|recommend|best treatment|clinical decision',
                r'what would you do|treatment approach|management plan'
            ],
            QueryIntent.DRUG_INTERACTION: [
                r'drug interaction|medication conflict|contraindication',
                r'can.*take.*with|interact with|adverse reaction'
            ],
            QueryIntent.DIAGNOSIS_SUPPORT: [
                r'differential diagnosis|possible diagnoses|what could cause',
                r'diagnostic workup|rule out|consider diagnosis'
            ],
            QueryIntent.TREATMENT_RECOMMENDATION: [
                r'treatment for|how to treat|therapy for|management of',
                r'best medication for|recommended dose|treatment protocol'
            ],
            QueryIntent.COHORT_ANALYSIS: [
                r'patients with.*and|cohort of|population with',
                r'group of patients|subset of patients|patient demographics'
            ]
        }
        
        self.entity_extractors = {
            'patient_id': r'patient\s*(?:id\s*)?(\d+)',
            'age': r'(\d+)\s*(?:year|yr|y\.o\.|years?\s*old)',
            'gender': r'\b(male|female|man|woman|boy|girl|m|f)\b',
            'medications': r'\b(metformin|warfarin|aspirin|lisinopril|amlodipine|atorvastatin|levothyroxine)\b',
            'conditions': r'\b(diabetes|hypertension|copd|asthma|depression|anxiety|cancer|stroke|mi|heart\s*failure)\b',
            'vital_signs': r'bp\s*(\d+/\d+)|hr\s*(\d+)|temp\s*(\d+\.?\d*)',
            'lab_values': r'\b(glucose|creatinine|bun|hemoglobin|hematocrit|wbc|platelets)\s*(?::|is|=)?\s*(\d+\.?\d*)',
            'dosages': r'(\d+\.?\d*)\s*(mg|mcg|g|ml|units?|iu)'
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Comprehensive query analysis"""
        query_lower = query.lower()
        
        # Detect intent
        detected_intent = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Determine complexity
        complexity = self._assess_complexity(query, entities)
        
        # Identify required data sources
        data_sources = self._identify_data_sources(detected_intent, entities)
        
        return {
            'intent': detected_intent,
            'entities': entities,
            'complexity': complexity,
            'data_sources': data_sources,
            'original_query': query,
            'processed_query': self._clean_query(query)
        }
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the primary intent of the query"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            scores[intent] = score
        
        # Return intent with highest score, default to literature search
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return QueryIntent.LITERATURE_SEARCH
    
    def _extract_entities(self, query: str) -> Dict:
        """Extract relevant medical entities from query"""
        entities = {}
        
        for entity_type, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if entity_type == 'patient_id':
                    entities[entity_type] = matches[0] if isinstance(matches[0], str) else matches[0]
                else:
                    entities[entity_type] = matches
        
        return entities
    
    def _assess_complexity(self, query: str, entities: Dict) -> str:
        """Assess query complexity for processing strategy"""
        score = 1  # Base complexity
        
        # Multiple entities increase complexity
        if len(entities) > 3:
            score += 1
        
        # Multiple questions in one query
        if len(re.findall(r'\?', query)) > 1:
            score += 1
        
        # Conditional logic (if/then, when, while)
        if re.search(r'\b(if|when|while|given that|assuming)\b', query, re.IGNORECASE):
            score += 1
        
        # Comparison requests
        if re.search(r'\b(compare|versus|vs|better than|worse than)\b', query, re.IGNORECASE):
            score += 1
        
        # Temporal elements
        if re.search(r'\b(before|after|during|timeline|progression)\b', query, re.IGNORECASE):
            score += 1
        
        if score <= 2:
            return 'simple'
        elif score <= 4:
            return 'medium'
        else:
            return 'complex'
    
    def _identify_data_sources(self, intent: QueryIntent, entities: Dict) -> List[str]:
        """Identify which data sources are needed"""
        sources = []
        
        # Always include literature for evidence-based responses
        if intent in [QueryIntent.LITERATURE_SEARCH, QueryIntent.CLINICAL_DECISION, 
                     QueryIntent.TREATMENT_RECOMMENDATION, QueryIntent.DIAGNOSIS_SUPPORT]:
            sources.append('literature')
        
        # Include patient data for patient-specific queries
        if intent in [QueryIntent.PATIENT_LOOKUP, QueryIntent.PATIENT_SIMILARITY, 
                     QueryIntent.COHORT_ANALYSIS]:
            sources.append('patients')
        
        # Include both for complex clinical decisions
        if intent == QueryIntent.CLINICAL_DECISION:
            sources.append('patients')
        
        # Special handling for drug interactions
        if intent == QueryIntent.DRUG_INTERACTION:
            sources.extend(['literature', 'drug_database'])
        
        # If patient ID is mentioned, always include patient data
        if 'patient_id' in entities:
            sources.append('patients')
        
        return list(set(sources))  # Remove duplicates
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Remove common filler words but preserve medical context
        cleaned = query.strip()
        
        # Normalize medical abbreviations
        medical_abbrevs = {
            r'\bpt\b': 'patient',
            r'\bdx\b': 'diagnosis',
            r'\btx\b': 'treatment',
            r'\bhtn\b': 'hypertension',
            r'\bdm\b': 'diabetes mellitus',
            r'\bmi\b': 'myocardial infarction',
            r'\bcad\b': 'coronary artery disease',
            r'\bcopd\b': 'chronic obstructive pulmonary disease',
            r'\bafib\b': 'atrial fibrillation'
        }
        
        for abbrev, full_form in medical_abbrevs.items():
            cleaned = re.sub(abbrev, full_form, cleaned, flags=re.IGNORECASE)
        
        return cleaned

class SmartQueryProcessor:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.router = QueryRouter()
        
    def process_complex_query(self, query: str) -> RAGResponse:
        """Process complex queries with advanced routing"""
        
        # Analyze query
        analysis = self.router.analyze_query(query)
        
        print(f"Query analysis: Intent={analysis['intent']}, Complexity={analysis['complexity']}")
        
        # Handle different intents with specialized processing
        if analysis['intent'] == QueryIntent.PATIENT_SIMILARITY:
            return self._handle_patient_similarity(analysis)
        elif analysis['intent'] == QueryIntent.DRUG_INTERACTION:
            return self._handle_drug_interaction(analysis)
        elif analysis['intent'] == QueryIntent.COHORT_ANALYSIS:
            return self._handle_cohort_analysis(analysis)
        else:
            # Use standard RAG processing
            return self.rag_system.process_query(analysis['processed_query'])
    
    def _handle_patient_similarity(self, analysis: Dict) -> RAGResponse:
        """Handle patient similarity queries"""
        if 'patient_id' in analysis['entities']:
            patient_id = str(analysis['entities']['patient_id'])
            
            try:
                # Find similar patients
                similar_patients = self.rag_system.patient_processor.find_similar_patients(
                    patient_id, {'diagnoses': True, 'medications': True}, n_similar=5
                )
                
                # Create response
                if similar_patients:
                    response_text = f"Found {len(similar_patients)} patients similar to patient {patient_id}:\n\n"
                    for i, patient in enumerate(similar_patients, 1):
                        response_text += f"{i}. Patient {patient['patient_id']} (similarity: {patient['similarity_score']:.2f})\n"
                        response_text += f"   Summary: {patient['summary'][:100]}...\n\n"
                else:
                    response_text = f"No similar patients found for patient {patient_id}. This could be due to unique characteristics or limited data availability."
                
                return RAGResponse(
                    answer=response_text,
                    sources=[{'type': 'patient', 'patient_id': p['patient_id']} for p in similar_patients],
                    confidence=0.85,
                    query_type='patient_similarity'
                )
            except Exception as e:
                return RAGResponse(
                    answer=f"Error retrieving similar patients: {str(e)}",
                    sources=[],
                    confidence=0.0,
                    query_type='error'
                )
        else:
            return self.rag_system.process_query(analysis['processed_query'])
    
    def _handle_drug_interaction(self, analysis: Dict) -> RAGResponse:
        """Handle drug interaction queries"""
        medications = analysis['entities'].get('medications', [])
        
        if len(medications) >= 2:
            # Search for drug interaction literature
            interaction_query = f"drug interaction {' and '.join(medications)}"
            return self.rag_system.process_query(interaction_query)
        else:
            # General drug interaction search
            return self.rag_system.process_query(analysis['processed_query'])
    
    def _handle_cohort_analysis(self, analysis: Dict) -> RAGResponse:
        """Handle cohort analysis queries"""
        # Extract cohort criteria from entities
        criteria = {}
        if 'age' in analysis['entities']:
            criteria['age_range'] = analysis['entities']['age']
        if 'gender' in analysis['entities']:
            criteria['gender'] = analysis['entities']['gender'][0]
        if 'conditions' in analysis['entities']:
            criteria['diagnoses'] = analysis['entities']['conditions']
        
        try:
            # Generate cohort
            cohort_patients = self.rag_system.patient_processor.generate_patient_cohort(criteria)
            
            response_text = f"Found cohort of {len(cohort_patients)} patients matching criteria:\n"
            response_text += f"Criteria: {criteria}\n"
            response_text += f"Patient IDs: {', '.join(cohort_patients[:10])}"
            if len(cohort_patients) > 10:
                response_text += f" and {len(cohort_patients) - 10} more..."
            
            return RAGResponse(
                answer=response_text,
                sources=[{'type': 'cohort', 'size': len(cohort_patients), 'criteria': criteria}],
                confidence=0.90,
                query_type='cohort_analysis'
            )
        except Exception as e:
            return RAGResponse(
                answer=f"Error generating patient cohort: {str(e)}",
                sources=[],
                confidence=0.0,
                query_type='error'
            )