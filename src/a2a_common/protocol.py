import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse

AUTH_HEADER_NAME = "X-Agent-Auth-Token"
SHARED_SECRET = "clinical-agentic-secure-token-2026"
api_key_header = APIKeyHeader(name=AUTH_HEADER_NAME, auto_error=False)

def verify_token(token: Optional[str] = Security(api_key_header)) -> bool:
    if not token or token != SHARED_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-Agent-Auth-Token"
        )
    return True

class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    framework: str # "LangGraph", "Google ADK", "Agno"
    port: int
    url: str
    primitives: List[str] = []
    streaming_supported: bool = False
    skills: List[Dict[str, Any]] = []
    capabilities: Dict[str, Any] = Field(default_factory=dict)

class A2AMessage(BaseModel):
    message_id: str
    patient_id: Optional[str] = None
    role: str = "user"
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False

class A2AResponse(BaseModel):
    message_id: str
    agent_name: str
    status: str = "completed"
    content: Any
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

def create_a2a_fastapi_app(agent_card: AgentCard) -> FastAPI:
    """Helper to create a standard FastAPI server with A2A Protocol endpoints."""
    app = FastAPI(
        title=f"A2A - {agent_card.name}",
        description=agent_card.description,
        version=agent_card.version
    )

    @app.get("/.well-known/agent.json")
    def get_agent_card():
        return agent_card.model_dump()

    @app.get("/health")
    def health():
        return {"status": "ok", "agent": agent_card.name, "port": agent_card.port}

    return app
