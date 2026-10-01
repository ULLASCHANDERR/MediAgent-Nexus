import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
import httpx

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.a2a_common.protocol import (
    AgentCard, A2AMessage, A2AResponse, create_a2a_fastapi_app, verify_token
)
from src.observability.tracker import tracker

# LangGraph State Schema
class ExtractorState(TypedDict):
    patient_id: str
    trace_id: str
    raw_discharge_text: str
    raw_lab_text: str
    bill_data: Dict[str, Any]
    language: str
    extracted_data: Dict[str, Any]
    status: str

# Node 1: Fetch raw document resources and prompts from Primary MCP Server
async def harvest_node(state: ExtractorState) -> Dict[str, Any]:
    patient_id = state["patient_id"]
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "extractor_harvest_node", input_data={"patient_id": patient_id})

    raw_discharge = state.get("raw_discharge_text", "")
    raw_lab = state.get("raw_lab_text", "")
    bill_data = state.get("bill_data", {})

    if not raw_discharge:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("http://localhost:8200/clinicaltools/tools/clinical_data_harvester", json={"patient_id": patient_id})
            if resp.status_code == 200:
                data = resp.json()
                raw_discharge = data.get("discharge_report_raw", "")
                raw_lab = data.get("lab_report_raw", "")
                bill_data = data.get("bill_data", {})

    tracker.end_span(span_id, output_data={"raw_len": len(raw_discharge)})
    return {
        "raw_discharge_text": raw_discharge,
        "raw_lab_text": raw_lab,
        "bill_data": bill_data,
        "status": "harvested"
    }

# Node 2: Extract structured clinical entities
async def parse_entities_node(state: ExtractorState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "extractor_parse_entities_node")
    discharge_text = state.get("raw_discharge_text", "")
    lab_text = state.get("raw_lab_text", "")
    bill = state.get("bill_data", {})

    extracted: Dict[str, Any] = {
        "patient_id": state["patient_id"],
        "patient_name": "",
        "age": 50,
        "gender": "Unknown",
        "address": "",
        "admission_date": "",
        "discharge_date": "",
        "ward": "",
        "bed_no": "",
        "attending_doctor": "",
        "discharge_diagnosis": "",
        "medications": [],
        "adr_allergy_info": "",
        "follow_up_appointments": [],
        "discharge_instructions": "",
        "discharge_approved_by": "",
        "discharge_approved": False,
        "labs": [],
        "bill": bill
    }

    # Extract demographic info using regex
    name_match = re.search(r"(?:Patient Name|Nombre del Paciente|Name des Patienten|रोगी का नाम)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if name_match:
        extracted["patient_name"] = name_match.group(1).strip()

    age_match = re.search(r"(?:Age|Edad|Alter|आयु)\s*:\s*(\d+)", discharge_text, re.IGNORECASE)
    if age_match:
        extracted["age"] = int(age_match.group(1))

    gender_match = re.search(r"(?:Gender|Género|Geschlecht|लिंग)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if gender_match:
        extracted["gender"] = gender_match.group(1).strip()

    adm_match = re.search(r"(?:Admission Date|Fecha de Admisión|Aufnahmedatum|भर्ती की तारीख)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if adm_match:
        extracted["admission_date"] = adm_match.group(1).strip()

    dis_match = re.search(r"(?:Discharge Date|Fecha de Alta|Entlassungsdatum|छुट्टी की तारीख)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if dis_match:
        extracted["discharge_date"] = dis_match.group(1).strip()

    ward_match = re.search(r"(?:Ward|Unidad|Station|वार्ड)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if ward_match:
        extracted["ward"] = ward_match.group(1).strip()

    bed_match = re.search(r"(?:Bed No|Cama|Bettnummer|बेड नंबर)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if bed_match:
        extracted["bed_no"] = bed_match.group(1).strip()

    doctor_match = re.search(r"(?:Attending Doctors?|Médicos? Tratantes?|Behandelnde Ärzte|उपचार करने वाले चिकित्सक)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if doctor_match:
        extracted["attending_doctor"] = doctor_match.group(1).strip()

    diag_match = re.search(r"(?:Discharge Diagnosis|Diagnóstico de Alta|Entlassungsdiagnose|डिस्चार्ज निदान)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if diag_match:
        extracted["discharge_diagnosis"] = diag_match.group(1).strip()

    allergy_match = re.search(r"(?:Allergies|Adverse Drug Reactions|Alergias Conocidas|Allergien und Unverträglichkeiten|दवाइयों से एलर्जी)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if allergy_match:
        extracted["adr_allergy_info"] = allergy_match.group(1).strip()

    # Extract medications (lines starting with number followed by med info)
    med_lines = re.findall(r"\d+\.\s*([^\n\r]+)", discharge_text)
    for line in med_lines:
        parts = [p.strip() for p in line.split(",")]
        if parts:
            name_strength = parts[0].split()
            med_name = name_strength[0] if name_strength else "Medication"
            strength = name_strength[1] if len(name_strength) > 1 else "Standard"
            dosage = parts[1] if len(parts) > 1 else "1 tablet"
            freq = parts[2] if len(parts) > 2 else "QD"
            extracted["medications"].append({
                "medicine_name": med_name,
                "strength": strength,
                "dosage": dosage,
                "frequency": freq,
                "route": "PO",
                "period": "30 days",
                "remarks": parts[-1] if len(parts) > 3 else "As directed",
                "total_quantity": "30 tablets"
            })

    # Approval check
    approved_match = re.search(r"(?:Discharge Approved|Alta Aprobada|Entlassung genehmigt|डिस्चार्ज स्वीकृत)\s*:\s*(Yes|True|Ja|Sí|हाँ)", discharge_text, re.IGNORECASE)
    if approved_match:
        extracted["discharge_approved"] = True

    approver_match = re.search(r"(?:Approved By|Aprobado por|Genehmigt durch|अनुमोदनकर्ता)\s*:\s*([^\n\r]+)", discharge_text, re.IGNORECASE)
    if approver_match:
        extracted["discharge_approved_by"] = approver_match.group(1).strip()

    # Instructions
    inst_match = re.search(r"(?:DISCHARGE INSTRUCTIONS|INSTRUCCIONES DE ALTA|ENTLASSUNGSHINWEISE|डिस्चार्ज निर्देश)\s*[\r\n]+([^\n\r]+)", discharge_text, re.IGNORECASE)
    if inst_match:
        extracted["discharge_instructions"] = inst_match.group(1).strip()

    # Follow-ups
    fu_match = re.search(r"(?:FOLLOW-UP APPOINTMENTS|CITAS DE SEGUIMIENTO|FOLGETERMINE|फॉलो-अप परामर्श)\s*[\r\n]+([^\n\r]+)", discharge_text, re.IGNORECASE)
    if fu_match:
        extracted["follow_up_appointments"] = [{"specialty": "Clinic Consult", "instructions": fu_match.group(1).strip()}]

    # Labs extraction
    lab_entries = re.findall(r"\d+\.\s*([^:]+):\s*([^\(\[\n\r]+)(?:\((?:Reference Range|Rango|Referenzbereich|Reference Range):\s*([^\)]+)\))?\s*(?:\[([^\]]+)\])?", lab_text)
    for entry in lab_entries:
        test_name = entry[0].strip()
        val_unit = entry[1].strip()
        ref = entry[2].strip() if len(entry) > 2 and entry[2] else "Standard Range"
        flag = entry[3].strip() if len(entry) > 3 and entry[3] else "Normal"
        flag_type = "high" if "HIGH" in flag or "HOCH" in flag or "CRÍTICO" in flag or "उच्च" in flag else "normal"
        extracted["labs"].append({
            "test_name": test_name,
            "result_value": val_unit,
            "unit": "",
            "reference_range": ref,
            "flag": flag,
            "flag_type": flag_type
        })

    tracker.end_span(span_id, output_data={"meds_count": len(extracted["medications"])})
    return {"extracted_data": extracted, "status": "extracted"}

# Build LangGraph workflow
def build_extractor_graph():
    workflow = StateGraph(ExtractorState)
    workflow.add_node("harvest", harvest_node)
    workflow.add_node("parse_entities", parse_entities_node)
    workflow.add_edge(START, "harvest")
    workflow.add_edge("harvest", "parse_entities")
    workflow.add_edge("parse_entities", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

extractor_graph = build_extractor_graph()

agent_card = AgentCard(
    name="Clinical Extractor Agent",
    description="LangGraph-powered agent extracting structured clinical data from discharge, lab, and billing files.",
    framework="LangGraph",
    port=8100,
    url="http://localhost:8100",
    primitives=["Tools", "Resources", "Prompts"],
    streaming_supported=False
)

app = create_a2a_fastapi_app(agent_card)

@app.post("/a2a/message", response_model=A2AResponse)
async def handle_message(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    patient_id = msg.patient_id or (msg.content.get("patient_id") if isinstance(msg.content, dict) else "P001")
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(patient_id)
    
    initial_state: ExtractorState = {
        "patient_id": patient_id,
        "trace_id": trace_id,
        "raw_discharge_text": "",
        "raw_lab_text": "",
        "bill_data": {},
        "language": "auto",
        "extracted_data": {},
        "status": "initiated"
    }

    config = {"configurable": {"thread_id": f"extractor-{patient_id}"}}
    result_state = await extractor_graph.ainvoke(initial_state, config=config)

    return A2AResponse(
        message_id=msg.message_id,
        agent_name=agent_card.name,
        status="completed",
        content=result_state["extracted_data"],
        trace_id=trace_id,
        metadata={"medications_count": len(result_state["extracted_data"].get("medications", []))}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.agents.extractor_agent:app", host="0.0.0.0", port=8100, reload=False)
