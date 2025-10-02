# src/data_processing/mimic_processor.py
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import os

class MIMICProcessor:
    def __init__(self, mimic_db_path: str, source: str = 'csv', csv_dir: Optional[str] = None):
        self.source = source
        self.db_path = mimic_db_path
        self.csv_dir = csv_dir or os.path.dirname(mimic_db_path)
        self.conn = None
        if self.source == 'sqlite':
            self.conn = sqlite3.connect(mimic_db_path)
        else:
            # Preload CSVs if present
            self._load_csvs()

    def _load_csvs(self):
        def read_optional(name: str) -> Optional[pd.DataFrame]:
            path = os.path.join(self.csv_dir, name)
            if os.path.exists(path):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return None
            return None
        self.df_patients = read_optional('patients.csv')
        self.df_admissions = read_optional('admissions.csv')
        self.df_diagnoses_icd = read_optional('diagnoses_icd.csv')
        self.df_d_icd = read_optional('d_icd_diagnoses.csv')
        self.df_prescriptions = read_optional('prescriptions.csv')
        self.df_noteevents = read_optional('noteevents.csv')
        
    def get_patient_demographics(self, patient_ids: List[str] = None) -> pd.DataFrame:
        """Extract patient demographics"""
        if self.source == 'sqlite' and self.conn is not None:
            query = """
            SELECT 
                subject_id,
                gender,
                anchor_age as age,
                anchor_year
            FROM patients
            """
            if patient_ids:
                placeholders = ','.join(['?' for _ in patient_ids])
                query += f" WHERE subject_id IN ({placeholders})"
                return pd.read_sql_query(query, self.conn, params=patient_ids)
            else:
                return pd.read_sql_query(query, self.conn)
        else:
            df = self.df_patients.copy() if self.df_patients is not None else pd.DataFrame()
            if df.empty:
                return pd.DataFrame()
            # Normalize column names if different
            rename_map = {
                'anchor_age': 'age',
            }
            df = df.rename(columns=rename_map)
            # Fallback: compute age if missing and birthdate present
            if 'age' not in df.columns:
                for birth_col in ['BIRTHDATE', 'birthdate', 'dob']:
                    if birth_col in df.columns:
                        try:
                            bd = pd.to_datetime(df[birth_col].astype(str).str[:10], errors='coerce')
                            today = pd.Timestamp.today()
                            df['age'] = (today.year - bd.dt.year - ((today.month, today.day) < (bd.dt.month, bd.dt.day))).clip(lower=0)
                        except Exception:
                            pass
                        break
            # Filter by patient_ids if provided
            if patient_ids:
                df = df[df.get('subject_id').astype(str).isin([str(x) for x in patient_ids])]
            # Build safe column list
            cols = ['subject_id']
            if 'gender' in df.columns:
                cols.append('gender')
            if 'age' in df.columns:
                cols.append('age')
            return df[cols].copy() if not df.empty else pd.DataFrame()
    
    def get_patient_diagnoses(self, patient_id: str) -> List[Dict]:
        """Get all diagnoses for a patient"""
        if self.source == 'sqlite' and self.conn is not None:
            query = """
            SELECT 
                d.icd_code,
                d.icd_version,
                di.long_title as diagnosis_description,
                a.admittime
            FROM diagnoses_icd d
            JOIN admissions a ON d.hadm_id = a.hadm_id
            JOIN d_icd_diagnoses di ON d.icd_code = di.icd_code
            WHERE d.subject_id = ?
            ORDER BY a.admittime DESC
            """
            result = pd.read_sql_query(query, self.conn, params=[patient_id])
            return result.to_dict('records')
        else:
            if self.df_diagnoses_icd is None or self.df_d_icd is None:
                return []
            df = self.df_diagnoses_icd.copy()
            df = df[df['subject_id'].astype(str) == str(patient_id)]
            df = df.merge(self.df_d_icd, on='icd_code', how='left')
            if self.df_admissions is not None and 'hadm_id' in self.df_admissions.columns:
                df = df.merge(self.df_admissions[['hadm_id','admittime']], on='hadm_id', how='left')
            df = df.rename(columns={'long_title':'diagnosis_description'})
            if 'admittime' in df.columns:
                try:
                    df['admittime'] = pd.to_datetime(df['admittime'])
                    df = df.sort_values('admittime', ascending=False)
                except Exception:
                    pass
            return df[['icd_code','icd_version','diagnosis_description','admittime']].to_dict('records') if not df.empty else []
    
    def get_patient_medications(self, patient_id: str) -> List[Dict]:
        """Get medications for a patient"""
        if self.source == 'sqlite' and self.conn is not None:
            query = """
            SELECT 
                drug,
                dose_val_rx as dose,
                dose_unit_rx as unit,
                starttime,
                endtime
            FROM prescriptions
            WHERE subject_id = ?
            ORDER BY starttime DESC
            """
            result = pd.read_sql_query(query, self.conn, params=[patient_id])
            return result.to_dict('records')
        else:
            if self.df_prescriptions is None:
                return []
            df = self.df_prescriptions.copy()
            # Normalize column names if your CSVs differ
            rename_map = {
                'dose_val_rx': 'dose',
                'dose_unit_rx': 'unit'
            }
            df = df.rename(columns=rename_map)
            df = df[df['subject_id'].astype(str) == str(patient_id)]
            if 'starttime' in df.columns:
                try:
                    df['starttime'] = pd.to_datetime(df['starttime'])
                    df = df.sort_values('starttime', ascending=True)
                except Exception:
                    pass
            wanted = ['drug','dose','unit','starttime','endtime']
            existing = [c for c in wanted if c in df.columns]
            return df[existing].to_dict('records') if not df.empty else []
    
    def get_patient_notes(self, patient_id: str, note_types: List[str] = None) -> List[Dict]:
        """Get clinical notes for a patient"""
        if self.source == 'sqlite' and self.conn is not None:
            query = """
            SELECT 
                category,
                description,
                text,
                charttime
            FROM noteevents
            WHERE subject_id = ?
            """
            params = [patient_id]
            if note_types:
                placeholders = ','.join(['?' for _ in note_types])
                query += f" AND category IN ({placeholders})"
                params.extend(note_types)
            query += " ORDER BY charttime DESC"
            result = pd.read_sql_query(query, self.conn, params=params)
            return result.to_dict('records')
        else:
            if self.df_noteevents is None:
                return []
            df = self.df_noteevents.copy()
            df = df[df['subject_id'].astype(str) == str(patient_id)]
            if note_types and 'category' in df.columns:
                df = df[df['category'].astype(str).isin(note_types)]
            if 'charttime' in df.columns:
                try:
                    df['charttime'] = pd.to_datetime(df['charttime'])
                    df = df.sort_values('charttime', ascending=False)
                except Exception:
                    pass
            wanted = ['category','description','text','charttime']
            existing = [c for c in wanted if c in df.columns]
            return df[existing].to_dict('records') if not df.empty else []
    
    def create_patient_summary(self, patient_id: str) -> Dict:
        """Create comprehensive patient summary"""
        demographics = self.get_patient_demographics([patient_id])
        diagnoses = self.get_patient_diagnoses(patient_id)
        medications = self.get_patient_medications(patient_id)
        notes = self.get_patient_notes(patient_id, ['Discharge summary', 'Physician ', 'Nursing'])
        
        return {
            'patient_id': patient_id,
            'demographics': demographics.to_dict('records')[0] if not demographics.empty else {},
            'diagnoses': diagnoses[:10],  # Limit to recent diagnoses
            'medications': medications[:20],  # Recent medications
            'notes': notes[:5],  # Recent notes
            'summary_text': self._create_text_summary(demographics, diagnoses, medications)
        }
    
    def _create_text_summary(self, demographics, diagnoses, medications) -> str:
        """Create a text summary for embedding"""
        summary_parts = []
        
        if not demographics.empty:
            demo = demographics.iloc[0]
            gender_val = demo.get('gender', '') if isinstance(demo, dict) else demo['gender'] if 'gender' in demographics.columns else ''
            age_val = demo.get('age', '') if isinstance(demo, dict) else (demo['age'] if 'age' in demographics.columns else '')
            summary_parts.append(f"Patient: {gender_val}{('' if age_val=='' else f', age {age_val}')}")
        
        if diagnoses:
            recent_dx = [dx['diagnosis_description'] for dx in diagnoses[:5]]
            summary_parts.append(f"Diagnoses: {', '.join(recent_dx)}")
            
        if medications:
            recent_meds = [med['drug'] for med in medications[:10]]
            summary_parts.append(f"Medications: {', '.join(recent_meds)}")
            
        return ". ".join(summary_parts)

# No example execution on import