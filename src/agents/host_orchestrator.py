import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
import gradio as gr
import httpx

from src.a2a_common.client import A2AClient
from src.a2a_common.protocol import A2AMessage
from src.observability.tracker import tracker

a2a_client = A2AClient()

PORTS = {
    "mock_ehr": "http://localhost:8050",
    "primary_mcp": "http://localhost:8200",
    "secondary_mcp": "http://localhost:8201",
    "extractor": "http://localhost:8100",
    "validator": "http://localhost:8101",
    "normalizer": "http://localhost:8102",
    "monitor": "http://localhost:8103",
    "summary": "http://localhost:8104",
    "rag": "http://localhost:8105"
}

async def orchestrate_patient_discharge(patient_id: str) -> Dict[str, Any]:
    trace_id = tracker.create_trace(patient_id)
    host_span = tracker.start_span(trace_id, "host_orchestrator_pipeline", input_data={"patient_id": patient_id})

    # Step 1: Monitor Agent
    mon_msg = A2AMessage(message_id=f"msg-{uuid.uuid4().hex[:6]}", patient_id=patient_id, content={}, metadata={"trace_id": trace_id})
    try:
        mon_resp = await a2a_client.send_message(PORTS["monitor"], mon_msg)
    except Exception as e:
        mon_resp = {"content": {"discovered_cases": []}}

    # Step 2: Extractor Agent
    ext_msg = A2AMessage(message_id=f"msg-{uuid.uuid4().hex[:6]}", patient_id=patient_id, content={"patient_id": patient_id}, metadata={"trace_id": trace_id})
    ext_resp = await a2a_client.send_message(PORTS["extractor"], ext_msg)
    extracted_data = ext_resp.get("content", {})

    # Step 3: Normalizer Agent
    norm_msg = A2AMessage(message_id=f"msg-{uuid.uuid4().hex[:6]}", patient_id=patient_id, content=extracted_data, metadata={"trace_id": trace_id})
    norm_resp = await a2a_client.send_message(PORTS["normalizer"], norm_msg)
    normalized_result = norm_resp.get("content", {})
    normalized_data = normalized_result.get("normalized_data", extracted_data)

    # Step 4: Validation Agent
    val_msg = A2AMessage(message_id=f"msg-{uuid.uuid4().hex[:6]}", patient_id=patient_id, content=normalized_data, metadata={"trace_id": trace_id})
    val_resp = await a2a_client.send_message(PORTS["validator"], val_msg)
    validation_report = val_resp.get("content", {})

    # Step 5: Primary MCP Reporter Tool
    reporter_payload = {
        "patient_id": patient_id,
        "discharge_data": normalized_data,
        "validation_result": validation_report,
        "risk_info": {
            "risk_level": validation_report.get("risk_level", "Low"),
            "composite_score": validation_report.get("composite_risk_score", 20.0)
        },
        "trace_id": trace_id
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        rep_resp = await client.post(f"{PORTS['primary_mcp']}/clinicaltools/tools/clinical_insight_reporter", json=reporter_payload)
        rep_data = rep_resp.json() if rep_resp.status_code == 200 else {}

    tracker.end_span(host_span, output_data={"is_blocked": validation_report.get("is_discharge_blocked", False)})

    return {
        "trace_id": trace_id,
        "patient_id": patient_id,
        "extracted_data": normalized_data,
        "validation_report": validation_report,
        "audit_report_json": rep_data.get("audit_report_json", {}),
        "rendered_html": rep_data.get("rendered_html_summary", "")
    }

def run_gradio_pipeline(patient_id: str):
    res = asyncio.run(orchestrate_patient_discharge(patient_id))
    report = res["validation_report"]
    status_summary = f"""### Pipeline Execution Complete!
- **Patient ID:** {res['patient_id']}
- **Trace ID:** `{res['trace_id']}`
- **Completeness Score:** {report.get('completeness_score', 0)}%
- **Risk Level:** **{report.get('risk_level', 'Low')}** (Score: {report.get('composite_risk_score', 0)}/100)
- **Status:** **{'🚨 DISCHARGE BLOCKED - HITL REQUIRED' if report.get('is_discharge_blocked') else '✅ APPROVED FOR DISCHARGE'}**
- **Critical Issues Found:** {len([i for i in report.get('issues', []) if i.get('severity') == 'Critical'])}
"""
    issues_text = json.dumps(report.get("issues", []), indent=2)
    audit_json_str = json.dumps(res["audit_report_json"], indent=2)
    return status_summary, issues_text, audit_json_str

def build_gradio_app():
    with gr.Blocks(title="Host Orchestrator - Clinical Multi-Agent System") as demo:
        gr.Markdown("# 🏥 Hospital Discharge AI - Host Orchestrator (Google ADK)")
        gr.Markdown("Orchestrates LangGraph, Google ADK, and Agno agents across A2A Protocol and Dual MCP Servers.")
        
        with gr.Row():
            with gr.Column(scale=1):
                patient_dropdown = gr.Dropdown(
                    choices=["P001", "P002", "P003", "P004"],
                    value="P001",
                    label="Select Patient Case"
                )
                trigger_btn = gr.Button("⚡ Run Full A2A Clinical Pipeline", variant="primary")
            with gr.Column(scale=2):
                status_output = gr.Markdown("Select a patient and click the button to trigger orchestrator.")

        with gr.Tabs():
            with gr.TabItem("Validation & Safety Issues"):
                issues_output = gr.Code(label="Discovered Clinical Issues", language="json")
            with gr.TabItem("Full Audit Report JSON"):
                audit_output = gr.Code(label="Audit Report", language="json")

        trigger_btn.click(
            fn=run_gradio_pipeline,
            inputs=[patient_dropdown],
            outputs=[status_summary if 'status_summary' in locals() else status_output, issues_output, audit_output]
        )
    return demo

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=8083, share=False)
