#!/usr/bin/env python3
"""
Medical RAG System - Main Application
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
from pathlib import Path
SRC_DIR = str((Path(__file__).resolve().parent / 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.settings import settings

# Ensure required directories exist BEFORE configuring logging
Path('logs').mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_setup():
    """Check if system is properly set up"""
    checks = []
    
    # Check API key
    if not settings.OPENAI_API_KEY:
        checks.append("❌ OpenAI API key not found in .env file")
    else:
        checks.append("✅ OpenAI API key configured")
    
    # Check data source
    if settings.MIMIC_SOURCE == 'sqlite':
        if not os.path.exists(settings.MIMIC_DB_PATH):
            checks.append("❌ MIMIC SQLite DB not found - set MIMIC_SOURCE=csv or run setup.py")
        else:
            checks.append("✅ MIMIC SQLite DB found")
    else:
        required_csvs = ['patients.csv', 'admissions.csv', 'diagnoses_icd.csv', 'd_icd_diagnoses.csv', 'prescriptions.csv', 'noteevents.csv']
        existing = []
        for name in required_csvs:
            path = os.path.join(settings.MIMIC_CSV_DIR, name)
            if os.path.exists(path):
                existing.append(name)
        if existing:
            checks.append(f"✅ CSV dataset found: {', '.join(existing)}")
        else:
            checks.append("❌ No CSV files found in data/mimic")
    
    # Check directories
    required_dirs = ['data/chroma_db', 'logs']
    for directory in required_dirs:
        if not os.path.exists(directory):
            checks.append(f"❌ Directory missing: {directory}")
        else:
            checks.append(f"✅ Directory found: {directory}")
    
    # Print results
    print("\n🔍 System Check:")
    print("-" * 30)
    for check in checks:
        print(check)
    
    # Return True if all checks pass
    return all("✅" in check for check in checks)

def initialize_system():
    """Initialize all system components"""
    logger.info("Initializing Medical RAG System...")
    
    try:
        # Import system components
        from embeddings.vector_store import MedicalVectorStore
        from data_processing.mimic_processor import MIMICProcessor
        from patient_data.patient_processor import PatientDataProcessor
        from rag.medical_rag import MedicalRAGSystem
        from query_preprocessing.query_router import SmartQueryProcessor
        # Optional: Privacy manager if implemented later
        class PrivacyManager:
            pass
        
        # Initialize components
        logger.info("Initializing vector store...")
        vector_store = MedicalVectorStore(persist_directory=settings.CHROMA_DB_PATH)
        
        logger.info("Initializing MIMIC processor...")
        mimic_processor = MIMICProcessor(
            mimic_db_path=settings.MIMIC_DB_PATH,
            source=settings.MIMIC_SOURCE,
            csv_dir=settings.MIMIC_CSV_DIR
        )
        
        logger.info("Initializing patient processor...")
        patient_processor = PatientDataProcessor(mimic_processor, vector_store)
        
        logger.info("Initializing privacy manager...")
        privacy_manager = PrivacyManager()
        
        logger.info("Initializing RAG system...")
        rag_system = MedicalRAGSystem(
            vector_store=vector_store,
            patient_processor=patient_processor,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info("Initializing query processor...")
        query_processor = SmartQueryProcessor(rag_system)
        
        logger.info("All components initialized successfully")
        
        return {
            'vector_store': vector_store,
            'mimic_processor': mimic_processor,
            'patient_processor': patient_processor,
            'privacy_manager': privacy_manager,
            'rag_system': rag_system,
            'query_processor': query_processor
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def ingest_sample_data(components):
    """Ingest sample medical literature"""
    logger.info("Ingesting sample medical literature...")
    
    try:
        from data_processing.medical_preprocessor import MedicalDocument
        
        sample_literature = [
            {
                'doc_id': 'sepsis_guidelines_2024',
                'title': 'Sepsis Treatment Guidelines 2024',
                'content': '''
                Sepsis is a life-threatening condition that arises when the body's response to infection 
                causes injury to its own tissues and organs. Early recognition and treatment are critical.
                
                Key Treatment Elements:
                1. Early antibiotic administration within 1 hour of recognition
                2. Adequate fluid resuscitation with crystalloids (30ml/kg within first 3 hours)
                3. Vasopressor support when fluid resuscitation is inadequate
                4. Source control measures when applicable
                5. Supportive care including mechanical ventilation if needed
                
                Recent studies indicate that adherence to these guidelines can reduce mortality 
                by 15-20%. The Surviving Sepsis Campaign emphasizes the importance of the 
                "hour-1 bundle" for optimal outcomes.
                ''',
                'doc_type': 'clinical_guideline',
                'metadata': {'year': 2024, 'specialty': 'critical_care', 'source': 'Surviving Sepsis Campaign'}
            },
            {
                'doc_id': 'diabetes_heart_failure_management',
                'title': 'Diabetes and Heart Failure: Integrated Management Approach',
                'content': '''
                Patients with both diabetes and heart failure require specialized management approaches.
                The combination of these conditions significantly increases cardiovascular risk.
                
                Management Strategies:
                1. SGLT2 inhibitors: First-line therapy showing significant cardiovascular benefits
                2. ACE inhibitors or ARBs: Remain cornerstone therapy for heart failure
                3. Metformin: Safe in stable heart failure with normal kidney function
                4. Beta-blockers: Evidence-based therapy for heart failure
                5. Lifestyle interventions: Diet, exercise, and weight management
                
                Recent clinical trials demonstrate that SGLT2 inhibitors reduce heart failure 
                hospitalizations by 30-35% and cardiovascular mortality by 15-20% in this population.
                Regular monitoring of kidney function and electrolytes is essential.
                ''',
                'doc_type': 'research_summary',
                'metadata': {'year': 2024, 'specialty': 'cardiology', 'evidence_level': 'high'}
            },
            {
                'doc_id': 'drug_interactions_warfarin',
                'title': 'Warfarin Drug Interactions: Clinical Management',
                'content': '''
                Warfarin has numerous clinically significant drug interactions due to its narrow 
                therapeutic window and metabolism via cytochrome P450 enzymes.
                
                Major Drug Interactions:
                1. Antibiotics: Can increase INR by altering gut flora
                2. NSAIDs: Increase bleeding risk through multiple mechanisms
                3. Amiodarone: Inhibits warfarin metabolism, requires dose reduction
                4. Rifampin: Induces warfarin metabolism, requires dose increase
                
                Metformin and Warfarin: No significant pharmacokinetic interaction exists between 
                metformin and warfarin. They can be safely co-administered with routine INR monitoring.
                Both drugs are eliminated primarily by the kidneys, so dose adjustments may be 
                needed in renal impairment.
                ''',
                'doc_type': 'clinical_reference',
                'metadata': {'year': 2024, 'specialty': 'pharmacy', 'interaction_level': 'low'}
            }
        ]
        
        # Convert to MedicalDocument objects
        documents = [MedicalDocument(**doc) for doc in sample_literature]
        
        # Add to vector store
        components['vector_store'].add_literature_documents(documents)
        
        logger.info("✅ Sample literature ingested successfully")
        
        # Try to ingest sample patient data
        try:
            sample_patient_ids = ['10001', '10002', '10003']
            patient_records = components['patient_processor'].create_patient_embedding_data(sample_patient_ids)
            components['vector_store'].add_patient_records(patient_records)
            logger.info(f"✅ {len(patient_records)} patient records ingested successfully")
        except Exception as e:
            logger.warning(f"Could not ingest patient data: {e}")
        
    except Exception as e:
        logger.error(f"Failed to ingest sample data: {e}")
        raise

def run_cli_interface(components):
    """Run command-line interface"""
    print("\n🏥 Medical RAG System - Interactive CLI")
    print("=" * 50)
    print("Ask questions about medical literature or patient data")
    print("Type 'help' for examples, 'quit' to exit")
    
    query_processor = components['query_processor']
    
    while True:
        print("\n" + "-" * 30)
        query = input("❓ Enter your medical question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using Medical RAG System!")
            break
        
        if query.lower() == 'help':
            print("\n📖 Example questions:")
            examples = [
                "What are the latest treatments for sepsis?",
                "Can warfarin interact with metformin?",
                "How should diabetes and heart failure be managed together?",
                "Find patients similar to patient 10001",
                "What medications are commonly used for heart failure?"
            ]
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example}")
            continue
        
        if not query:
            print("Please enter a valid question.")
            continue
        
        try:
            print("\n🔍 Processing your query...")
            response = query_processor.process_complex_query(query)
            
            print(f"\n📊 Query Analysis:")
            print(f"   Type: {response.query_type.replace('_', ' ').title()}")
            print(f"   Confidence: {response.confidence:.1%}")
            
            print(f"\n💡 Answer:")
            print("-" * 20)
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources, 1):
                    if source['type'] == 'literature':
                        print(f"   {i}. 📄 {source.get('title', 'Medical Literature')}")
                    elif source['type'] == 'patient':
                        print(f"   {i}. 👤 Patient {source.get('patient_id', 'Unknown')}")
                    else:
                        print(f"   {i}. 📊 {source['type'].title()}")
                        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}")
            print("Please try rephrasing your question.")

def run_streamlit_interface():
    """Run Streamlit web interface"""
    try:
        import streamlit
        print("🌐 Starting Streamlit web interface...")
        print("Open your browser to: http://localhost:8501")
        
        # Import and run Streamlit app
        from streamlit_app import create_streamlit_app
        create_streamlit_app()
        
    except ImportError:
        print("❌ Streamlit not available. Install with: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"Failed to start Streamlit: {e}")
        return False

def main():
    """Main application entry point"""
    print("🏥 Medical RAG System")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please run: python setup.py")
        return 1
    
    try:
        # Initialize system
        components = initialize_system()
        
        # Ingest sample data if vector store is empty
        if not os.path.exists(os.path.join(settings.CHROMA_DB_PATH, "chroma.sqlite3")):
            print("\n📚 Vector store is empty. Ingesting sample data...")
            ingest_sample_data(components)
        
        # Choose interface
        if len(sys.argv) > 1 and sys.argv[1] == '--web':
            run_streamlit_interface()
        else:
            run_cli_interface(components)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    # Ensure required directories exist
    Path('logs').mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    sys.exit(main())