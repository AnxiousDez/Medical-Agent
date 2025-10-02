import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# --- Configuration ---
NUM_PATIENTS = 1000
MAX_ADMISSIONS_PER_PATIENT = 3

# Initialize Faker for generating realistic data
fake = Faker()

# Define common lab item IDs and their properties (mimicking D_LABITEMS)
lab_item_definitions = {
    50912: {"label": "Creatinine", "unit": "mg/dL", "normal_low": 0.6, "normal_high": 1.2},
    50971: {"label": "Potassium", "unit": "mEq/L", "normal_low": 3.5, "normal_high": 5.0},
    50902: {"label": "Bilirubin", "unit": "mg/dL", "normal_low": 0.1, "normal_high": 1.2},
    50882: {"label": "Bicarbonate", "unit": "mEq/L", "normal_low": 22, "normal_high": 29},
    51221: {"label": "Hematocrit", "unit": "%", "normal_low": 36, "normal_high": 48},
    51301: {"label": "WBC", "unit": "K/uL", "normal_low": 4.5, "normal_high": 11.0},
}

# Define admission diagnoses templates
admission_diagnoses = [
    "Pneumonia", "Congestive Heart Failure Exacerbation", "Sepsis secondary to UTI",
    "Post-operative complications", "Gastrointestinal Bleed", "Acute Kidney Injury",
    "Diabetic Ketoacidosis", "COPD Exacerbation"
]

# --- Data Generation Functions ---

def create_patients(n):
    """Generates a DataFrame of synthetic patients."""
    patient_list = []
    for i in range(n):
        patient_id = 100000 + i
        gender = random.choice(['M', 'F'])
        dob = fake.date_of_birth(minimum_age=40, maximum_age=90)
        # Simulate realistic mortality rate
        dod = None
        if random.random() < 0.2: # 20% mortality rate overall
            dod = fake.date_between(start_date=dob + timedelta(days=365*40), end_date=datetime.now().date())
        
        patient_list.append({
            "subject_id": patient_id,
            "gender": gender,
            "dob": dob,
            "dod": dod
        })
    return pd.DataFrame(patient_list)

def create_admissions(patients_df):
    """Generates admissions for the given patients."""
    admission_list = []
    hadm_id_counter = 500000
    for _, patient in patients_df.iterrows():
        num_admissions = random.randint(1, MAX_ADMISSIONS_PER_PATIENT)
        for _ in range(num_admissions):
            min_admit_date = patient['dob'] + timedelta(days=365*18)
            max_admit_date = patient['dod'] if pd.notna(patient['dod']) else datetime.now().date()

            if max_admit_date <= min_admit_date:
                continue

            admit_date_only = fake.date_between(start_date=min_admit_date, end_date=max_admit_date)
            admittime = datetime.combine(admit_date_only, fake.time_object())

            admission_duration_days = random.randint(2, 14)
            dischtime = admittime + timedelta(days=admission_duration_days, hours=random.randint(1, 23))
            
            # --- FIX IS HERE ---
            # Ensure discharge time does not exceed date of death by comparing two datetime objects.
            if pd.notna(patient['dod']):
                # Convert patient['dod'] from a date object to a datetime object (at midnight)
                dod_datetime = datetime.combine(patient['dod'], datetime.min.time())
                
                # Now compare two datetime objects to ensure consistency
                if dischtime > dod_datetime:
                    dischtime = dod_datetime
            # --- End Fix ---

            admission_list.append({
                "subject_id": patient["subject_id"],
                "hadm_id": hadm_id_counter,
                "admittime": admittime,
                "dischtime": dischtime,
                "diagnosis": random.choice(admission_diagnoses),
                "ethnicity": random.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO", "OTHER"]),
            })
            hadm_id_counter += 1
    return pd.DataFrame(admission_list)

def create_labevents(admissions_df):
    """Generates synthetic lab events for each admission."""
    lab_list = []
    for _, admission in admissions_df.iterrows():
        # Generate between 5 and 20 lab panels per admission
        for _ in range(random.randint(5, 20)):
            itemid = random.choice(list(lab_item_definitions.keys()))
            item_details = lab_item_definitions[itemid]
            
            if random.random() < 0.8:
                value = round(random.uniform(item_details["normal_low"], item_details["normal_high"]), 1)
            else:
                value = round(random.uniform(item_details["normal_high"], item_details["normal_high"] * 1.5), 1)

            # Ensure lab time is within admission boundaries, handle potential edge case where admittime == dischtime
            try:
                lab_time = fake.date_time_between(start_date=admission["admittime"], end_date=admission["dischtime"])
            except ValueError:
                lab_time = admission["admittime"] # Default to admission time if range is invalid
            
            lab_list.append({
                "subject_id": admission["subject_id"],
                "hadm_id": admission["hadm_id"],
                "itemid": itemid,
                "charttime": lab_time,
                "value": value,
                "valuenum": value,
                "valueuom": item_details["unit"],
                "label": item_details["label"]
            })
    return pd.DataFrame(lab_list)

def create_noteevents(admissions_df, patients_df):
    """Generates synthetic discharge summaries for each admission."""
    note_list = []
    patients_dict = patients_df.set_index('subject_id').to_dict('index')
    
    for _, admission in admissions_df.iterrows():
        patient = patients_dict[admission["subject_id"]]
        age = admission["admittime"].year - patient["dob"].year
        diagnosis = admission["diagnosis"]
        
        history_illness = f"Patient is a {age}-year-old {patient['gender']} with a past medical history significant for hypertension and diabetes mellitus, who presented to the emergency department with complaints of shortness of breath and fever for three days."
        hospital_course = f"The patient was admitted for {diagnosis}. Initial labs were significant for elevated WBC and creatinine. Chest X-ray revealed a right lower lobe infiltrate. Patient was started on intravenous antibiotics (Ceftriaxone) and hydration. Patient responded well to therapy, with normalization of vital signs and improvement in symptoms. Repeat labs showed resolution of leukocytosis and improvement in renal function. Patient was deemed stable for discharge."
        discharge_plan = "Discharge Condition: Stable. Follow up with Primary Care Physician in 1 week. Continue oral antibiotics for 7 days. Patient advised to monitor blood glucose levels closely."
        
        full_note = f"""
Admission Date: {admission['admittime'].strftime('%Y-%m-%d %H:%M')}
Discharge Date: {admission['dischtime'].strftime('%Y-%m-%d %H:%M')}

HISTORY OF PRESENT ILLNESS:
{history_illness}

HOSPITAL COURSE:
{hospital_course}

DISCHARGE PLAN:
{discharge_plan}
"""
        note_list.append({
            "subject_id": admission["subject_id"],
            "hadm_id": admission["hadm_id"],
            "chartdate": admission["dischtime"].date(),
            "category": "Discharge summary",
            "text": full_note
        })
    return pd.DataFrame(note_list)

# --- Main execution ---
if __name__ == "__main__":
    print(f"Generating data for {NUM_PATIENTS} patients...")
    
    df_patients = create_patients(NUM_PATIENTS)
    df_patients.to_csv("patients.csv", index=False)
    print(f"Generated {len(df_patients)} patient records.")

    df_admissions = create_admissions(df_patients)
    df_admissions.to_csv("admissions.csv", index=False)
    print(f"Generated {len(df_admissions)} admission records.")

    if not df_admissions.empty:
        df_labs = create_labevents(df_admissions)
        df_labs.to_csv("labevents.csv", index=False)
        print(f"Generated {len(df_labs)} lab events.")

        df_notes = create_noteevents(df_admissions, df_patients)
        df_notes.to_csv("noteevents.csv", index=False)
        print(f"Generated {len(df_notes)} clinical notes.")
    else:
        print("No admissions generated, skipping labs and notes.")
    
    print("\nCSV files created successfully!")