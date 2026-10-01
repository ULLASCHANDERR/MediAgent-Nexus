import os
import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Secondary Analytics MCP Server",
    description="Exposes clinical analytics, risk scoring, population benchmarks, and risk heatmaps.",
    version="1.0.0"
)

# Benchmark database
POPULATION_BENCHMARKS = {
    "Type 2 Diabetes Mellitus with Acute Hyperglycemia": {
        "30_day_readmission_rate_pct": 14.2,
        "avg_length_of_stay_days": 4.8,
        "medication_adherence_gap_pct": 22.0,
        "post_discharge_complication_risk": "Moderate",
        "benchmark_sample_size": 18450
    },
    "Acute Coronary Syndrome / Non-ST Elevation Myocardial Infarction (NSTEMI)": {
        "30_day_readmission_rate_pct": 19.8,
        "avg_length_of_stay_days": 6.2,
        "medication_adherence_gap_pct": 18.5,
        "post_discharge_complication_risk": "High",
        "benchmark_sample_size": 12300
    },
    "Bacterial Pneumonia": {
        "30_day_readmission_rate_pct": 16.5,
        "avg_length_of_stay_days": 5.4,
        "medication_adherence_gap_pct": 15.0,
        "post_discharge_complication_risk": "Moderate",
        "benchmark_sample_size": 14200
    },
    "Hypertensive Crisis with Mild Renal Impairment": {
        "30_day_readmission_rate_pct": 17.1,
        "avg_length_of_stay_days": 5.1,
        "medication_adherence_gap_pct": 28.4,
        "post_discharge_complication_risk": "High",
        "benchmark_sample_size": 9800
    },
    "Default": {
        "30_day_readmission_rate_pct": 15.0,
        "avg_length_of_stay_days": 5.0,
        "medication_adherence_gap_pct": 20.0,
        "post_discharge_complication_risk": "Moderate",
        "benchmark_sample_size": 50000
    }
}

@app.get("/analyticstools/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "calculate_risk_score",
                "description": "Calculates composite 0-100 discharge risk score and risk level (Low/Medium/High)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                        "diagnosis": {"type": "string"},
                        "medications_count": {"type": "integer"},
                        "critical_issues_count": {"type": "integer"},
                        "is_bill_paid": {"type": "boolean"},
                        "has_abnormal_labs": {"type": "boolean"}
                    }
                }
            },
            {
                "name": "get_population_benchmarks",
                "description": "Retrieves national readmission rates and benchmarks by clinical diagnosis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "diagnosis": {"type": "string"}
                    }
                }
            },
            {
                "name": "generate_risk_heatmap",
                "description": "Returns multi-dimensional clinical and operational risk scores",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "composite_score": {"type": "number"}
                    }
                }
            }
        ]
    }

@app.post("/analyticstools/tools/calculate_risk_score")
def calculate_risk_score(payload: Dict[str, Any]):
    age = payload.get("age", 50)
    med_count = payload.get("medications_count", 3)
    critical_issues = payload.get("critical_issues_count", 0)
    is_bill_paid = payload.get("is_bill_paid", True)
    has_abnormal_labs = payload.get("has_abnormal_labs", False)

    # Base risk score calculation
    score = 15.0

    # Age factor
    if age > 60:
        score += 15.0
    elif age > 45:
        score += 8.0

    # Polypharmacy factor
    if med_count >= 4:
        score += 20.0
    elif med_count >= 2:
        score += 10.0

    # Abnormal labs
    if has_abnormal_labs:
        score += 15.0

    # Critical cross-validation issues
    if critical_issues > 0:
        score += (critical_issues * 25.0)

    # Unsettled financial status
    if not is_bill_paid:
        score += 15.0

    # Cap at 100
    final_score = min(100.0, round(score, 1))

    if final_score >= 60.0 or critical_issues > 0:
        risk_level = "High"
    elif final_score >= 35.0:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "status": "success",
        "composite_score": final_score,
        "risk_level": risk_level,
        "factors": {
            "age_contribution": 15 if age > 60 else (8 if age > 45 else 0),
            "medication_burden": 20 if med_count >= 4 else 10,
            "critical_validation_penalty": critical_issues * 25,
            "financial_unsettled_penalty": 0 if is_bill_paid else 15
        }
    }

@app.post("/analyticstools/tools/get_population_benchmarks")
def get_population_benchmarks(payload: Dict[str, Any]):
    diagnosis = payload.get("diagnosis", "")
    matched_benchmark = POPULATION_BENCHMARKS.get("Default")
    
    for diag_key in POPULATION_BENCHMARKS:
        if diag_key.lower() in diagnosis.lower() or diagnosis.lower() in diag_key.lower():
            matched_benchmark = POPULATION_BENCHMARKS[diag_key]
            break

    return {
        "status": "success",
        "diagnosis": diagnosis,
        "benchmarks": matched_benchmark
    }

@app.post("/analyticstools/tools/generate_risk_heatmap")
def generate_risk_heatmap(payload: Dict[str, Any]):
    composite = payload.get("composite_score", 40.0)
    readmission_subscore = min(100.0, round(composite * 1.1, 1))
    interaction_subscore = min(100.0, round(composite * 0.9, 1))
    adherence_subscore = min(100.0, round(composite * 0.85, 1))
    payer_subscore = min(100.0, round(composite * 0.95, 1))

    return {
        "status": "success",
        "patient_id": payload.get("patient_id", "P000"),
        "dimensions": [
            {"dimension": "30-Day Readmission Risk", "score": readmission_subscore, "category": "Clinical"},
            {"dimension": "Drug-Drug Interaction Risk", "score": interaction_subscore, "category": "Pharmacology"},
            {"dimension": "Patient Adherence & Health Literacy", "score": adherence_subscore, "category": "Behavioral"},
            {"dimension": "Payer & Billing Settlement Risk", "score": payer_subscore, "category": "Operational"}
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "Secondary MCP Analytics Server", "port": 8201}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.mcp_servers.secondary_analytics_server:app", host="0.0.0.0", port=8201, reload=False)
