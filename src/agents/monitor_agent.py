import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException
import httpx

from src.a2a_common.protocol import (
    AgentCard, A2AMessage, A2AResponse, create_a2a_fastapi_app, verify_token
)
from src.observability.tracker import tracker

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_ROOT_URI = f"file://{BASE_DIR / 'data' / 'input'}"

class GoogleADKMonitorAgent:
    """Discharge Monitor Agent built with Google ADK patterns, integrating MCP Roots."""
    def __init__(self, root_uri: str = INPUT_ROOT_URI):
        self.root_uri = root_uri
        self.name = "Discharge Monitor Agent"

    async def scan_incoming_patients(self, trace_id: str) -> Dict[str, Any]:
        span_id = tracker.start_span(trace_id, "monitor_scan_roots", input_data={"root_uri": self.root_uri})
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Discover authorized roots from Primary MCP Server
            roots_resp = await client.get("http://localhost:8200/clinicaltools/roots")
            authorized_roots = roots_resp.json().get("roots", []) if roots_resp.status_code == 200 else []

            # 2. Invoke Clinical Watcher Tool with authorized Root URI
            tool_resp = await client.post("http://localhost:8200/clinicaltools/tools/clinical_watcher", json={
                "root_uri": self.root_uri
            })
            watcher_res = tool_resp.json() if tool_resp.status_code == 200 else {"patients_discovered": []}

        tracker.end_span(span_id, output_data={"patients_count": watcher_res.get("total_patients", 0)})
        return {
            "root_uri": self.root_uri,
            "authorized_roots": authorized_roots,
            "discovered_cases": watcher_res.get("patients_discovered", []),
            "total_cases": watcher_res.get("total_patients", 0)
        }

monitor_adk_agent = GoogleADKMonitorAgent()

agent_card = AgentCard(
    name="Discharge Monitor Agent",
    description="Google ADK agent monitoring clinical document repository via MCP Roots with path traversal security.",
    framework="Google ADK",
    port=8103,
    url="http://localhost:8103",
    primitives=["Tools", "Roots"],
    streaming_supported=False
)

app = create_a2a_fastapi_app(agent_card)

@app.post("/a2a/message", response_model=A2AResponse)
async def handle_message(msg: A2AMessage, authorized: bool = Depends(verify_token)):
    trace_id = msg.metadata.get("trace_id") or tracker.create_trace(msg.patient_id or "ROOT_MONITOR")
    result = await monitor_adk_agent.scan_incoming_patients(trace_id)

    return A2AResponse(
        message_id=msg.message_id,
        agent_name=agent_card.name,
        status="completed",
        content=result,
        trace_id=trace_id,
        metadata={"cases_found": result["total_cases"]}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.agents.monitor_agent:app", host="0.0.0.0", port=8103, reload=False)
