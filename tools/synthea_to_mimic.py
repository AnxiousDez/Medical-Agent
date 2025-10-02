import os
import pandas as pd
from pathlib import Path
from datetime import datetime

SRC = Path("data/synthea")
DST = Path("data/mimic")
DST.mkdir(parents=True, exist_ok=True)

def read_csv(name: str) -> pd.DataFrame:
    path = SRC / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

patients = read_csv("patients.csv")
encounters = read_csv("encounters.csv")
conditions = read_csv("conditions.csv")
meds = read_csv("medications.csv")

# 1) patients.csv -> subject_id, gender, anchor_age
if not patients.empty:
    def compute_age(birthdate):
        try:
            bd = datetime.fromisoformat(str(birthdate)[:10])
            today = datetime.now()
            return max(0, today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day)))
        except Exception:
            return None
    out = pd.DataFrame({
        "subject_id": patients.get("Id", patients.get("ID")),
        "gender": patients.get("GENDER"),
        "anchor_age": patients.get("BIRTHDATE").apply(compute_age) if "BIRTHDATE" in patients.columns else None
    })
    out.to_csv(DST / "patients.csv", index=False)

# 2) admissions.csv -> hadm_id, subject_id, admittime, dischtime
if not encounters.empty:
    adm = pd.DataFrame({
        "hadm_id": encounters.get("Id", encounters.get("ID")),
        "subject_id": encounters.get("PATIENT"),
        "admittime": encounters.get("START"),
        "dischtime": encounters.get("STOP")
    })
    adm.to_csv(DST / "admissions.csv", index=False)

# 3) diagnoses_icd.csv & d_icd_diagnoses.csv
if not conditions.empty:
    di = pd.DataFrame({
        "hadm_id": conditions.get("ENCOUNTER"),
        "subject_id": conditions.get("PATIENT"),
        "icd_code": conditions.get("CODE"),
        "icd_version": 10
    })
    di.to_csv(DST / "diagnoses_icd.csv", index=False)

    dx_ref = conditions[["CODE", "DESCRIPTION"]].dropna().drop_duplicates()
    dx_ref.columns = ["icd_code", "long_title"]
    dx_ref.to_csv(DST / "d_icd_diagnoses.csv", index=False)

# 4) prescriptions.csv -> subject_id, hadm_id, drug, dose_val_rx, dose_unit_rx, starttime, endtime
if not meds.empty:
    pr = pd.DataFrame({
        "subject_id": meds.get("PATIENT"),
        "hadm_id": meds.get("ENCOUNTER"),
        "drug": meds.get("DESCRIPTION"),
        "dose_val_rx": "",
        "dose_unit_rx": "",
        "starttime": meds.get("START"),
        "endtime": meds.get("STOP")
    })
    pr.to_csv(DST / "prescriptions.csv", index=False)

# 5) noteevents.csv (synthetic discharge notes)
note_rows = []
if not encounters.empty:
    cond_by_enc = conditions.groupby("ENCOUNTER") if not conditions.empty else None
    meds_by_enc = meds.groupby("ENCOUNTER") if not meds.empty else None
    for _, enc in encounters.iterrows():
        hadm = enc.get("Id", enc.get("ID"))
        subj = enc.get("PATIENT")
        diag_list = []
        if cond_by_enc is not None:
            try:
                diag_list = cond_by_enc.get_group(hadm)["DESCRIPTION"].dropna().unique().tolist()
            except Exception:
                pass
        med_list = []
        if meds_by_enc is not None:
            try:
                med_list = meds_by_enc.get_group(hadm)["DESCRIPTION"].dropna().unique().tolist()
            except Exception:
                pass
        text = f"Discharge Summary for patient {subj}. Diagnoses: {', '.join(diag_list[:5])}. Medications: {', '.join(med_list[:8])}."
        note_rows.append({
            "subject_id": subj,
            "hadm_id": hadm,
            "category": "Discharge summary",
            "text": text,
            "charttime": enc.get("STOP", enc.get("START", ""))
        })
if note_rows:
    pd.DataFrame(note_rows).to_csv(DST / "noteevents.csv", index=False)

print("✓ Wrote MIMIC-like CSVs to data/mimic")


