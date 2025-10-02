#!/usr/bin/env python3
"""
Setup script for Medical RAG System
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/literature",
        "data/mimic",
        "data/processed", 
        "data/chroma_db",
        "logs",
        "tests",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import chromadb
        import sentence_transformers
        import openai
        import spacy
        print("✓ Core dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def download_models():
    """Download required models"""
    try:
        # Download spaCy models (optional)
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        except subprocess.CalledProcessError as e:
            print("⚠️  Could not install en_core_web_sm automatically. Continuing without it.")
            print(f"   Details: {e}")
        
        # Download medical model (if not already installed)
        try:
            import spacy
            nlp = spacy.load("en_core_sci_sm")
            print("✓ Medical spaCy model found")
        except OSError:
            print("⚠️  SciSpaCy model not installed. Skipping (optional).")
        
        print("✓ Model step completed (optional)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error during model step: {e}. Continuing without models.")
        return True

def create_sample_mimic_db():
    """Create a sample MIMIC database for testing"""
    db_path = "data/mimic/mimic.db"
    
    if os.path.exists(db_path):
        print("✓ MIMIC database already exists")
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create sample tables (simplified MIMIC structure)
        cursor.execute('''
        CREATE TABLE patients (
            subject_id INTEGER PRIMARY KEY,
            gender TEXT,
            anchor_age INTEGER,
            anchor_year INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE admissions (
            hadm_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            admittime TEXT,
            dischtime TEXT,
            admission_type TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE diagnoses_icd (
            hadm_id INTEGER,
            subject_id INTEGER,
            icd_code TEXT,
            icd_version INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE d_icd_diagnoses (
            icd_code TEXT PRIMARY KEY,
            long_title TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE prescriptions (
            subject_id INTEGER,
            hadm_id INTEGER,
            drug TEXT,
            dose_val_rx REAL,
            dose_unit_rx TEXT,
            starttime TEXT,
            endtime TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE noteevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            category TEXT,
            description TEXT,
            text TEXT,
            charttime TEXT
        )
        ''')
        
        # Insert sample data
        sample_patients = [
            (10001, 'M', 65, 2020),
            (10002, 'F', 45, 2021),
            (10003, 'M', 78, 2019)
        ]
        cursor.executemany("INSERT INTO patients VALUES (?, ?, ?, ?)", sample_patients)
        
        sample_diagnoses = [
            ('E11', 'Type 2 diabetes mellitus'),
            ('I50', 'Heart failure'),
            ('J44', 'Chronic obstructive pulmonary disease')
        ]
        cursor.executemany("INSERT INTO d_icd_diagnoses VALUES (?, ?)", sample_diagnoses)
        
        sample_patient_diagnoses = [
            (1, 10001, 'E11', 10),
            (1, 10001, 'I50', 10),
            (2, 10002, 'J44', 10)
        ]
        cursor.executemany("INSERT INTO diagnoses_icd VALUES (?, ?, ?, ?)", sample_patient_diagnoses)
        
        conn.commit()
        conn.close()
        
        print("✓ Sample MIMIC database created")
        return True
    except Exception as e:
        print(f"✗ Error creating MIMIC database: {e}")
        return False

def main():
    """Main setup function"""
    print("🏥 Medical RAG System Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("Please install dependencies with: pip install -r requirements.txt")
        return False
    
    # Download models
    print("\n3. Downloading models...")
    if not download_models():
        return False
    
    # Create sample database
    print("\n4. Setting up sample database...")
    create_sample_mimic_db()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. If you have real MIMIC data, replace the sample database")
    print("3. Run: python main.py")
    
    return True

if __name__ == "__main__":
    main()