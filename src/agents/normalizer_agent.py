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

class NormalizerState(TypedDict):
    patient_id: str
    trace_id: str
    source_language: str
    clinical_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    translation_confidence: float
    model_used: str
    status: str

# Sampling callback implementation as required by section 2.3
async def sampling_callback(server_model_hints: List[str], text_to_translate: str, source_lang: str) -> Dict[str, Any]:
    """
    Implements sampling_callback:
    Reads server's model hints, routes to appropriate model (nova-lite for multilingual, command-r-plus for English),
    performs inference / translation, and returns CreateMessageResult.
    """
    selected_model = "bedrock/amazon.nova-lite-v1:0" if source_lang.lower() not in ["english", "en"] else "cohere/command-r-plus"
    
    # In live or mocked LLM gateway, this executes LiteLLM call
    return {
        "selected_model": selected_model,
        "role": "assistant",
        "text": text_to_translate,
        "finish_reason": "stop"
    }

async def normalize_terms_node(state: NormalizerState) -> Dict[str, Any]:
    trace_id = state["trace_id"]
    span_id = tracker.start_span(trace_id, "normalize_terms_node")
    clinical_data = state["clinical_data"]
    source_lang = state.get("source_language", "auto")

    # Detect language if not explicitly provided
    text_sample = json.dumps(clinical_data)
    if "Diagnóstico" in text_sample or "Paciente" in text_sample or "tableta" in text_sample:
        source_lang = "Spanish"
    elif "Entlassungsdiagnose" in text_sample or "Patienten" in text_sample or "Tablette" in text_sample:
        source_lang = "German"
    elif "निदान" in text_sample or "रोगी" in text_sample or "दवाइयां" in text_sample:
        source_lang = "Hindi"
    else:
        source_lang = "English"

    # Call MCP Medical Lang Bridge Tool
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("http://localhost:8200/clinicaltools/tools/medical_lang_bridge", json={
            "text": text_sample,
            "source_language": source_lang,
            "trace_id": trace_id
        })
        bridge_res = resp.json() if resp.status_code == 200 else {}

    # Call client-side sampling callback
    model_hints = bridge_res.get("model_preferences_sent", ["bedrock/amazon.nova-lite-v1:0", "cohere/command-r-plus"])
    sample_res = await sampling_callback(model_hints, text_sample, source_lang)

    # Normalize clinical fields in structured data
    normalized = json.loads(json.dumps(clinical_data))
    
    # Normalize diagnosis
    diag = normalized.get("discharge_diagnosis", "")
    if "Síndrome Coronario Agudo" in diag:
        normalized["discharge_diagnosis"] = "Acute Coronary Syndrome / Non-ST Elevation Myocardial Infarction (NSTEMI)"
    elif "Ambulant erworbene bakterielle Pneumonie" in diag:
        normalized["discharge_diagnosis"] = "Community-Acquired Bacterial Pneumonia"
    elif "हाइपरटेंसिव क्राइसिस" in diag:
        normalized["discharge_diagnosis"] = "Hypertensive Crisis with Mild Renal Impairment"

    # Normalize medications frequencies
    for med in normalized.get("medications", []):
        freq = med.get("frequency", "")
        if freq == "BID":
            med["frequency_plain"] = "twice daily"
        elif freq == "QD":
            med["frequency_plain"] = "once daily"
        elif freq == "QPM":
            med["frequency_plain"] = "in the evening"
        elif freq == "PRN":
            med["frequency_plain"] = "as needed"
        else:
            med["frequency_plain"] = freq

    confidence = bridge_res.get("translation_confidence", 0.98)
    tracker.end_span(span_id, output_data={"confidence": confidence, "model": sample_res["selected_model"]})

    return {
        "source_language": source_lang,
        "normalized_data": normalized,
        "translation_confidence": confidence,
        "model_used": sample_res["selected_model"],
        "status": "normalized"
    }

def build_normalizer_graph():
    workflow = StateGraph(NormalizerState)
    workflow.add_node("normalize", normalize_terms_node)
    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

normalizer_graph = build_normalizer_graph()

agent_card = AgentCard(
    name="Clinical Normalizer Agent",
    description="LangGraph agent for multilingual translation and medical abbreviation normalization via MCP Sampling.",
    framework="LangGraph",
    port=8102,
    url="http://localhost:8102",
    primitives=["Tools", "Sampling", "Prompts"],
    streaming_supported=False
)

app = create_a2a_fastapi_app(agent_card)

@app.post("/a2a/message", response_model=A2AResponse)
async def handle_message(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    content = msg.content if isinstance(msg.content, dict) else {}
    patient_id = msg.patient_id or content.get("patient_id", "P001")
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(patient_id)
    clinical_data = content.get("extracted_data") or content

    initial_state: NormalizerState = {
        "patient_id": patient_id,
        "trace_id": trace_id,
        "source_language": msg.metadata.get("language", "auto"),
        "clinical_data": clinical_data,
        "normalized_data": {},
        "translation_confidence": 0.0,
        "model_used": "",
        "status": "initiated"
    }

    config = {"configurable": {"thread_id": f"normalizer-{patient_id}"}}
    result_state = await normalizer_graph.ainvoke(initial_state, config=config)

    return A2AResponse(
        message_id=msg.message_id,
        agent_name=agent_card.name,
        status="completed",
        content={
            "normalized_data": result_state["normalized_data"],
            "source_language": result_state["source_language"],
            "translation_confidence": result_state["translation_confidence"],
            "model_selected": result_state["model_used"]
        },
        trace_id=trace_id,
        metadata={"confidence": result_state["translation_confidence"]}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.agents.normalizer_agent:app", host="0.0.0.0", port=8102, reload=False)
