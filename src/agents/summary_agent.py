import asyncio
import json
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from src.a2a_common.protocol import (
    AgentCard, A2AMessage, A2AResponse, create_a2a_fastapi_app, verify_token
)
from src.observability.tracker import tracker

class GoogleADKSummaryGenerator:
    """Discharge Summary Generator built with Google ADK patterns, featuring section-by-section A2A Streaming."""
    
    def __init__(self):
        self.name = "Discharge Summary Generator"

    async def generate_sections_stream(self, patient_id: str, clinical_data: Dict[str, Any], trace_id: str) -> AsyncGenerator[str, None]:
        span_id = tracker.start_span(trace_id, "summary_streaming_generation", input_data={"patient_id": patient_id})

        # Fetch prompt from Primary MCP Server
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                p_resp = await client.get("http://localhost:8200/clinicaltools/prompts/summary-generation-prompt")
                prompt_info = p_resp.json()
            except Exception:
                prompt_info = {}

        # Progressive Section 1: Patient Demographics & Hospital Stay
        sec1 = {
            "section": "patient_overview",
            "title": "1. Patient Demographics & Hospital Stay",
            "content": f"Patient **{clinical_data.get('patient_name', 'Patient')}** (ID: {patient_id}), {clinical_data.get('age', 50)}yo {clinical_data.get('gender', '')}, admitted on {clinical_data.get('admission_date', '2026-09-20')} to {clinical_data.get('ward', 'Inpatient Unit')} and discharged on {clinical_data.get('discharge_date', '2026-09-28')}. Primary Attending: {clinical_data.get('attending_doctor', 'Attending Physician')}."
        }
        yield f"data: {json.dumps(sec1)}\n\n"
        await asyncio.sleep(0.4)

        # Progressive Section 2: Diagnosis & Clinical Course
        sec2 = {
            "section": "diagnosis_course",
            "title": "2. Primary Diagnosis & What It Means",
            "content": f"**Discharge Diagnosis:** {clinical_data.get('discharge_diagnosis', 'Subacute Condition')}.\n\n*Clinical Note:* The acute exacerbation has responded favorably to therapeutic intervention. Vital signs and metabolic parameters are stable at the time of discharge."
        }
        yield f"data: {json.dumps(sec2)}\n\n"
        await asyncio.sleep(0.4)

        # Progressive Section 3: Medications & Plain-English Schedule
        meds = clinical_data.get("medications", [])
        med_lines = []
        for m in meds:
            med_lines.append(f"- **{m.get('medicine_name')}** ({m.get('strength')}): Take {m.get('dosage')} {m.get('frequency_plain', m.get('frequency'))} via {m.get('route')}. *Note:* {m.get('remarks')}")
        sec3 = {
            "section": "medications",
            "title": "3. Discharge Medications & Home Schedule",
            "content": "\n".join(med_lines) if med_lines else "No prescription changes required upon discharge."
        }
        yield f"data: {json.dumps(sec3)}\n\n"
        await asyncio.sleep(0.4)

        # Progressive Section 4: Key Laboratory Results
        labs = clinical_data.get("labs", [])
        lab_lines = []
        for l in labs:
            lab_lines.append(f"- **{l.get('test_name')}**: {l.get('result_value')} {l.get('unit')} (Reference: {l.get('reference_range')}) [{l.get('flag')}]")
        sec4 = {
            "section": "labs",
            "title": "4. Key Laboratory Results",
            "content": "\n".join(lab_lines) if lab_lines else "Routine laboratory values reviewed and filed."
        }
        yield f"data: {json.dumps(sec4)}\n\n"
        await asyncio.sleep(0.4)

        # Progressive Section 5: Financial Clearance & Billing
        bill = clinical_data.get("bill", {})
        sec5 = {
            "section": "billing",
            "title": "5. Billing & Financial Clearance",
            "content": f"Invoice: {bill.get('invoice_number', 'N/A')} | Total Charges: ${bill.get('total_amount', 0):,.2f} | Status: **{bill.get('payment_status', 'PAID')}**."
        }
        yield f"data: {json.dumps(sec5)}\n\n"
        await asyncio.sleep(0.4)

        # Progressive Section 6: Discharge Instructions & Follow-up
        sec6 = {
            "section": "instructions",
            "title": "6. Discharge Instructions & Red Flags",
            "content": f"{clinical_data.get('discharge_instructions', 'Follow routine convalescent care.')}\n\n**Warning Signs:** If you experience acute chest pain, shortness of breath, sudden fever, or extreme dizziness, proceed to the nearest emergency department immediately."
        }
        yield f"data: {json.dumps(sec6)}\n\n"
        await asyncio.sleep(0.2)

        tracker.end_span(span_id, output_data={"sections_emitted": 6})
        yield "data: [DONE]\n\n"

summary_generator = GoogleADKSummaryGenerator()

agent_card = AgentCard(
    name="Discharge Summary Generator",
    description="Google ADK agent providing progressive section-by-section streaming of patient discharge summaries.",
    framework="Google ADK",
    port=8104,
    url="http://localhost:8104",
    primitives=["Tools", "Prompts"],
    streaming_supported=True
)

app = create_a2a_fastapi_app(agent_card)

@app.post("/a2a/message/stream")
async def handle_stream(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    content = msg.content if isinstance(msg.content, dict) else {}
    patient_id = msg.patient_id or content.get("patient_id", "P001")
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(patient_id)
    clinical_data = content.get("clinical_data") or content

    return StreamingResponse(
        summary_generator.generate_sections_stream(patient_id, clinical_data, trace_id),
        media_type="text/event-stream"
    )

@app.post("/a2a/message", response_model=A2AResponse)
async def handle_non_streaming(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    content = msg.content if isinstance(msg.content, dict) else {}
    patient_id = msg.patient_id or content.get("patient_id", "P001")
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(patient_id)
    
    # Collect all sections
    sections = []
    async for chunk in summary_generator.generate_sections_stream(patient_id, content, trace_id):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            sections.append(json.loads(chunk[6:].strip()))

    return A2AResponse(
        message_id=msg.message_id,
        agent_name=agent_card.name,
        status="completed",
        content={"sections": sections},
        trace_id=trace_id,
        metadata={"sections_count": len(sections)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.agents.summary_agent:app", host="0.0.0.0", port=8104, reload=False)
