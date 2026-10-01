import json
from typing import Dict, Any, List, Optional, TypedDict
from fastapi import FastAPI, Depends, HTTPException
import httpx

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.a2a_common.protocol import (
    AgentCard, A2AMessage, A2AResponse, create_a2a_fastapi_app, verify_token
)
from src.observability.tracker import tracker
from src.guardrails.rai_guardrails import GuardrailManager

guardrails = GuardrailManager()

class ValidationState(TypedDict):
    patient_id: str
    trace_id: str
    extracted_data: Dict[str, Any]
    completeness_result: Dict[str, Any]
    ehr_cross_validation_result: Dict[str, Any]
    risk_info: Dict[str, Any]
    guardrail_decision: Dict[str, Any]
    elicitation_response: Optional[Dict[str, Any]]
    final_validation_report: Dict[str, Any]

# Node 1: Clinical Completeness Validation (MCP Rules Engine Tool)
async def completeness_validation_node(state: ValidationState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "completeness_validation_node")
    
    extracted = state["extracted_data"]
    payload = {
        "data": extracted,
        "doc_type": "discharge_report",
        "trace_id": trace_id,
        "elicitation_response": state.get("elicitation_response")
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("http://localhost:8200/clinicaltools/tools/clinical_rules_engine", json=payload)
        comp_res = resp.json() if resp.status_code == 200 else {
            "completeness_score": 90, "is_blocked": False, "missing_blocking_fields": []
        }

    tracker.end_span(span_id, output_data={"completeness_score": comp_res.get("completeness_score")})
    return {"completeness_result": comp_res}

# Node 2: EHR Cross-Validation (MCP EHR Validation Tool)
async def ehr_cross_validation_node(state: ValidationState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "ehr_cross_validation_node")
    patient_id = state["patient_id"]
    extracted = state["extracted_data"]

    payload = {
        "patient_id": patient_id,
        "discharge_data": extracted,
        "trace_id": trace_id
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("http://localhost:8200/clinicaltools/tools/ehr_validation", json=payload)
        ehr_res = resp.json() if resp.status_code == 200 else {
            "is_blocked": False, "issues": []
        }

    tracker.end_span(span_id, output_data={"issues_count": len(ehr_res.get("issues", []))})
    return {"ehr_cross_validation_result": ehr_res}

# Node 3: Risk Scoring & Analytics (Secondary MCP Server :8201)
async def analytics_node(state: ValidationState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "analytics_node")
    extracted = state["extracted_data"]
    comp_res = state["completeness_result"]
    ehr_res = state["ehr_cross_validation_result"]

    critical_count = len([i for i in ehr_res.get("issues", []) if i.get("severity") == "Critical"])
    if comp_res.get("is_blocked"):
        critical_count += 1

    bill = extracted.get("bill", {})
    is_bill_paid = (bill.get("payment_status") == "PAID") or bill.get("insurance_guarantee_letter", False)

    calc_payload = {
        "age": extracted.get("age", 50),
        "diagnosis": extracted.get("discharge_diagnosis", ""),
        "medications_count": len(extracted.get("medications", [])),
        "critical_issues_count": critical_count,
        "is_bill_paid": is_bill_paid,
        "has_abnormal_labs": any(lab.get("flag_type") == "high" for lab in extracted.get("labs", []))
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("http://localhost:8201/analyticstools/tools/calculate_risk_score", json=calc_payload)
        risk_res = resp.json() if resp.status_code == 200 else {
            "composite_score": 30.0, "risk_level": "Low"
        }

    tracker.end_span(span_id, output_data={"risk_level": risk_res.get("risk_level")})
    return {"risk_info": risk_res}

# Node 4: RAI Guardrails & Final Report Aggregation
async def guardrail_and_aggregate_node(state: ValidationState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "guardrail_and_aggregate_node")

    comp_res = state["completeness_result"]
    ehr_res = state["ehr_cross_validation_result"]
    risk_info = state["risk_info"]

    all_issues = ehr_res.get("issues", [])
    if comp_res.get("missing_blocking_fields"):
        for field in comp_res["missing_blocking_fields"]:
            all_issues.append({
                "rule_id": "clinical_completeness_check",
                "severity": "Critical",
                "description": f"Mandatory clinical field '{field}' is missing from discharge document",
                "action": "Block discharge"
            })

    is_discharge_blocked = comp_res.get("is_blocked", False) or ehr_res.get("is_blocked", False)

    validation_payload = {
        "risk_level": risk_info.get("risk_level", "Low"),
        "discharge_blocked": is_discharge_blocked,
        "issues": all_issues
    }

    guardrail_decision = guardrails.evaluate_discharge_case(validation_payload)

    tracker.log_guardrail_event(
        trace_id=trace_id,
        guardrail_name="GuardrailManager.evaluate_discharge_case",
        check_result=guardrail_decision["action"],
        action_taken="ESCALATE" if guardrail_decision["escalation_required"] else "PASS",
        blocked=guardrail_decision["discharge_blocked"],
        details=guardrail_decision["reasons"]
    )

    final_report = {
        "patient_id": state["patient_id"],
        "trace_id": trace_id,
        "completeness_score": comp_res.get("completeness_score", 100),
        "is_discharge_blocked": is_discharge_blocked,
        "risk_level": risk_info.get("risk_level", "Low"),
        "composite_risk_score": risk_info.get("composite_score", 20.0),
        "recommendation": "Reject / HITL Required" if is_discharge_blocked else "Approve Discharge",
        "issues": all_issues,
        "elicit_request": comp_res.get("elicit_request"),
        "guardrail_escalation": guardrail_decision
    }

    tracker.end_span(span_id, output_data={"action": guardrail_decision["action"]})
    return {
        "guardrail_decision": guardrail_decision,
        "final_validation_report": final_report
    }

def build_validation_graph():
    workflow = StateGraph(ValidationState)
    workflow.add_node("completeness", completeness_validation_node)
    workflow.add_node("cross_validation", ehr_cross_validation_node)
    workflow.add_node("analytics", analytics_node)
    workflow.add_node("guardrails", guardrail_and_aggregate_node)

    workflow.add_edge(START, "completeness")
    workflow.add_edge("completeness", "cross_validation")
    workflow.add_edge("cross_validation", "analytics")
    workflow.add_edge("analytics", "guardrails")
    workflow.add_edge("guardrails", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

validation_graph = build_validation_graph()

agent_card = AgentCard(
    name="Clinical Validation Agent",
    description="LangGraph-powered validation agent verifying clinical completeness, cross-referencing Mock EHR, and evaluating RAI guardrails.",
    framework="LangGraph",
    port=8101,
    url="http://localhost:8101",
    primitives=["Tools", "Elicitation", "Resources"],
    streaming_supported=False
)

app = create_a2a_fastapi_app(agent_card)

@app.post("/a2a/message", response_model=A2AResponse)
async def handle_message(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    content = msg.content if isinstance(msg.content, dict) else {}
    patient_id = msg.patient_id or content.get("patient_id", "P001")
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(patient_id)
    extracted_data = content.get("extracted_data") or content

    initial_state: ValidationState = {
        "patient_id": patient_id,
        "trace_id": trace_id,
        "extracted_data": extracted_data,
        "completeness_result": {},
        "ehr_cross_validation_result": {},
        "risk_info": {},
        "guardrail_decision": {},
        "elicitation_response": msg.metadata.get("elicitation_response"),
        "final_validation_report": {}
    }

    config = {"configurable": {"thread_id": f"validation-{patient_id}"}}
    result_state = await validation_graph.ainvoke(initial_state, config=config)

    return A2AResponse(
        message_id=msg.message_id,
        agent_name=agent_card.name,
        status="completed",
        content=result_state["final_validation_report"],
        trace_id=trace_id,
        metadata={"is_blocked": result_state["final_validation_report"]["is_discharge_blocked"]}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.agents.validation_agent:app", host="0.0.0.0", port=8101, reload=False)
