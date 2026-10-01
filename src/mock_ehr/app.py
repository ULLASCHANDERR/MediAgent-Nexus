import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Mock Hospital Electronic Health Record (EHR) API",
    description="Simulates hospital EHR database providing patient demographics, med history, allergies, labs, and care plans.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MOCK_DATA_DIR = BASE_DIR / "data" / "mock_ehr"

def load_json(filename: str):
    filepath = MOCK_DATA_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "service": "Mock EHR System", "port": 8050}

@app.get("/patients")
def get_all_patients():
    return load_json("patients.json")

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    patients = load_json("patients.json")
    for p in patients:
        if p.get("patient_id") == patient_id:
            return p
    raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in EHR")

@app.get("/patients/{patient_id}/medications")
def get_medications(patient_id: str):
    meds = load_json("medications.json")
    if patient_id not in meds:
        raise HTTPException(status_code=404, detail=f"Medications for patient {patient_id} not found")
    return meds[patient_id]

@app.get("/patients/{patient_id}/allergies")
def get_allergies(patient_id: str):
    allergies = load_json("allergies.json")
    return allergies.get(patient_id, [])

@app.get("/patients/{patient_id}/labs")
def get_labs(patient_id: str):
    labs = load_json("labs.json")
    if patient_id not in labs:
        raise HTTPException(status_code=404, detail=f"Lab records for patient {patient_id} not found")
    return labs[patient_id]

@app.get("/patients/{patient_id}/care_plan")
def get_care_plan(patient_id: str):
    care_plans = load_json("care_plans.json")
    if patient_id not in care_plans:
        raise HTTPException(status_code=404, detail=f"Care plan for patient {patient_id} not found")
    return care_plans[patient_id]

class MedicationCheckRequest(BaseModel):
    medications: List[Dict[str, Any]]

@app.post("/patients/{patient_id}/verify_medication")
def verify_medications(patient_id: str, req: MedicationCheckRequest):
    allergies = load_json("allergies.json").get(patient_id, [])
    conflicts = []
    
    for med in req.medications:
        med_name = med.get("medicine_name", "").lower()
        for allergy in allergies:
            allergy_substance = allergy.get("substance", "").lower()
            if allergy_substance in med_name or med_name in allergy_substance:
                conflicts.append({
                    "medicine_name": med.get("medicine_name"),
                    "allergy_substance": allergy.get("substance"),
                    "severity": allergy.get("severity"),
                    "reaction": allergy.get("reaction")
                })
                
    return {
        "patient_id": patient_id,
        "has_allergy_conflict": len(conflicts) > 0,
        "conflicts": conflicts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.mock_ehr.app:app", host="0.0.0.0", port=8050, reload=False)
