import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

random.seed(7)

MIMIC_DIR = Path('data/mimic')
MIMIC_DIR.mkdir(parents=True, exist_ok=True)

# Load existing CSVs (keep schema)
patients = pd.read_csv(MIMIC_DIR / 'patients.csv') if (MIMIC_DIR / 'patients.csv').exists() else pd.DataFrame()
admissions = pd.read_csv(MIMIC_DIR / 'admissions.csv') if (MIMIC_DIR / 'admissions.csv').exists() else pd.DataFrame()
diagnoses_icd = pd.read_csv(MIMIC_DIR / 'diagnoses_icd.csv') if (MIMIC_DIR / 'diagnoses_icd.csv').exists() else pd.DataFrame()
dx_dict = pd.read_csv(MIMIC_DIR / 'd_icd_diagnoses.csv') if (MIMIC_DIR / 'd_icd_diagnoses.csv').exists() else pd.DataFrame(columns=['icd_code','long_title'])
prescriptions = pd.read_csv(MIMIC_DIR / 'prescriptions.csv') if (MIMIC_DIR / 'prescriptions.csv').exists() else pd.DataFrame(columns=['subject_id','hadm_id','drug','dose_val_rx','dose_unit_rx','starttime','endtime'])
notes = pd.read_csv(MIMIC_DIR / 'noteevents.csv') if (MIMIC_DIR / 'noteevents.csv').exists() else pd.DataFrame(columns=['subject_id','hadm_id','category','text','charttime'])

# Catalogs to enrich data
NEW_DX = [
    ('I21', 'Acute myocardial infarction', 10),
    ('J18.9', 'Pneumonia, unspecified organism', 10),
    ('K92.2', 'Gastrointestinal hemorrhage, unspecified', 10),
    ('N17.9', 'Acute kidney failure, unspecified', 10),
    ('E78.5', 'Hyperlipidemia, unspecified', 10),
    ('G20', 'Parkinson disease', 10),
    ('C50.9', 'Malignant neoplasm of breast', 10),
]

NEW_MEDS = [
    'ceftriaxone','piperacillin-tazobactam','vancomycin','heparin','insulin regular',
    'metoprolol','rosuvastatin','omeprazole','amoxicillin-clavulanate','spironolactone'
]

def parse_dt(s):
    try:
        return datetime.fromisoformat(str(s).replace('T',' ').split('.')[0])
    except Exception:
        return datetime.now()

def fmt_dt(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def next_ids():
    next_subject = (patients['subject_id'].max() + 1) if 'subject_id' in patients.columns and not patients.empty else 100000
    next_hadm = (admissions['hadm_id'].max() + 1) if 'hadm_id' in admissions.columns and not admissions.empty else 500000
    return int(next_subject), int(next_hadm)

def ensure_dx_dict(codes):
    existing = set(dx_dict['icd_code']) if not dx_dict.empty else set()
    rows = []
    for code, title, _ in codes:
        if code not in existing:
            rows.append({'icd_code': code, 'long_title': title})
    return pd.concat([dx_dict, pd.DataFrame(rows)], ignore_index=True) if rows else dx_dict

def add_patients(n=300):
    start_id, _ = next_ids()
    rows = []
    for i in range(n):
        sid = start_id + i
        gender = random.choice(['M','F'])
        dob = datetime(1930,1,1) + timedelta(days=random.randint(0, 365*70))
        dod = '' if random.random() > 0.1 else (dob + timedelta(days=random.randint(365*40, 365*85))).date()
        rows.append({'subject_id': sid, 'gender': gender, 'dob': str(dob.date()), 'dod': str(dod) if dod else ''})
    return pd.concat([patients, pd.DataFrame(rows)], ignore_index=True)

def add_admissions_for(pat_df, max_per=3):
    _, next_hadm = next_ids()
    rows = []
    for _, p in pat_df.iterrows():
        for _ in range(random.randint(1, max_per)):
            adm = datetime(2019,1,1) + timedelta(days=random.randint(0, 365*5), hours=random.randint(0,23))
            dis = adm + timedelta(days=random.randint(2,10), hours=random.randint(0,23))
            diagnosis = random.choice(['Sepsis secondary to UTI','NSTEMI','Pneumonia','COPD Exacerbation','DKA','GI Bleed','AKI'])
            ethnicity = random.choice(['WHITE','BLACK/AFRICAN AMERICAN','ASIAN','HISPANIC/LATINO','OTHER'])
            rows.append({'subject_id': int(p['subject_id']), 'hadm_id': next_hadm, 'admittime': fmt_dt(adm), 'dischtime': fmt_dt(dis), 'diagnosis': diagnosis, 'ethnicity': ethnicity})
            next_hadm += 1
    return pd.concat([admissions, pd.DataFrame(rows)], ignore_index=True)

def add_diagnoses(adm_df):
    rows = []
    for _, a in adm_df.iterrows():
        for code, title, ver in random.sample(NEW_DX, k=random.randint(1,3)):
            rows.append({'hadm_id': int(a['hadm_id']), 'subject_id': int(a['subject_id']), 'icd_code': code, 'icd_version': ver})
    base = diagnoses_icd if not diagnoses_icd.empty else pd.DataFrame(columns=['hadm_id','subject_id','icd_code','icd_version'])
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)

def add_prescriptions(adm_df):
    rows = []
    for _, a in adm_df.iterrows():
        start = parse_dt(a['admittime']) + timedelta(days=random.randint(0,2))
        for _ in range(random.randint(1,4)):
            drug = random.choice(NEW_MEDS)
            end = start + timedelta(days=random.randint(2,7))
            rows.append({'subject_id': int(a['subject_id']), 'hadm_id': int(a['hadm_id']), 'drug': drug, 'dose_val_rx': '', 'dose_unit_rx': '', 'starttime': fmt_dt(start), 'endtime': fmt_dt(end)})
    base = prescriptions if not prescriptions.empty else pd.DataFrame(columns=['subject_id','hadm_id','drug','dose_val_rx','dose_unit_rx','starttime','endtime'])
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)

def add_notes(adm_df, dx_df, rx_df):
    rows = []
    dx_by_hadm = dx_df.groupby('hadm_id') if not dx_df.empty else None
    rx_by_hadm = rx_df.groupby('hadm_id') if not rx_df.empty else None
    for _, a in adm_df.iterrows():
        hadm = int(a['hadm_id'])
        dx_list = []
        if dx_by_hadm is not None:
            try:
                dx_list = dx_by_hadm.get_group(hadm)['icd_code'].tolist()
            except Exception:
                pass
        rx_list = []
        if rx_by_hadm is not None:
            try:
                rx_list = rx_by_hadm.get_group(hadm)['drug'].tolist()
            except Exception:
                pass
        text = (
            f"Admission Date: {a['admittime']}\n"
            f"Discharge Date: {a['dischtime']}\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "Patient admitted with symptoms suggestive of acute illness.\n\n"
            "HOSPITAL COURSE:\n"
            f"Diagnoses: {', '.join(dx_list)}.\n"
            f"Medications: {', '.join(rx_list)}.\n\n"
            "DISCHARGE PLAN:\nFollow up with PCP in 1 week."
        )
        rows.append({'subject_id': int(a['subject_id']), 'hadm_id': hadm, 'category': 'Discharge summary', 'text': text, 'charttime': a['dischtime'][:10]})
    base = notes if not notes.empty else pd.DataFrame(columns=['subject_id','hadm_id','category','text','charttime'])
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)

def main():
    # 1) Add patients
    new_patients = add_patients(n=500)
    # 2) Add admissions
    new_adm = add_admissions_for(new_patients[new_patients['subject_id'] >= new_patients['subject_id'].max() - 499])
    # 3) Add diagnoses and dict
    new_dx = add_diagnoses(new_adm[new_adm['hadm_id'] >= new_adm['hadm_id'].max() - (len(new_patients)*3)])
    new_dict = ensure_dx_dict(NEW_DX)
    # 4) Add prescriptions
    new_rx = add_prescriptions(new_adm[new_adm['hadm_id'] >= new_adm['hadm_id'].max() - (len(new_patients)*3)])
    # 5) Add notes
    new_notes = add_notes(new_adm[new_adm['hadm_id'] >= new_adm['hadm_id'].max() - (len(new_patients)*3)], new_dx, new_rx)

    # Write back preserving schema
    new_patients.to_csv(MIMIC_DIR / 'patients.csv', index=False)
    new_adm.to_csv(MIMIC_DIR / 'admissions.csv', index=False)
    new_dx.to_csv(MIMIC_DIR / 'diagnoses_icd.csv', index=False)
    new_dict.to_csv(MIMIC_DIR / 'd_icd_diagnoses.csv', index=False)
    new_rx.to_csv(MIMIC_DIR / 'prescriptions.csv', index=False)
    new_notes.to_csv(MIMIC_DIR / 'noteevents.csv', index=False)
    print('✓ Augmented patients, admissions, diagnoses, prescriptions, and notes in data/mimic')

if __name__ == '__main__':
    main()


