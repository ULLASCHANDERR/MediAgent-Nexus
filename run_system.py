import subprocess
import sys
import time
import signal
import os
import httpx
from typing import Dict, List

SERVICES = [
    {"name": "Mock EHR System", "port": 8050, "cmd": [sys.executable, "-m", "uvicorn", "src.mock_ehr.app:app", "--port", "8050", "--host", "0.0.0.0"]},
    {"name": "Primary MCP Clinical Tools", "port": 8200, "cmd": [sys.executable, "-m", "uvicorn", "src.mcp_servers.primary_clinical_server:app", "--port", "8200", "--host", "0.0.0.0"]},
    {"name": "Secondary MCP Analytics", "port": 8201, "cmd": [sys.executable, "-m", "uvicorn", "src.mcp_servers.secondary_analytics_server:app", "--port", "8201", "--host", "0.0.0.0"]},
    {"name": "Clinical Extractor Agent (LangGraph)", "port": 8100, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.extractor_agent:app", "--port", "8100", "--host", "0.0.0.0"]},
    {"name": "Clinical Validation Agent (LangGraph)", "port": 8101, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.validation_agent:app", "--port", "8101", "--host", "0.0.0.0"]},
    {"name": "Clinical Normalizer Agent (LangGraph)", "port": 8102, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.normalizer_agent:app", "--port", "8102", "--host", "0.0.0.0"]},
    {"name": "Discharge Monitor Agent (Google ADK)", "port": 8103, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.monitor_agent:app", "--port", "8103", "--host", "0.0.0.0"]},
    {"name": "Summary Generator Agent (Google ADK Streaming)", "port": 8104, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.summary_agent:app", "--port", "8104", "--host", "0.0.0.0"]},
    {"name": "Clinical RAG Q&A Agent (Agno Streaming)", "port": 8105, "cmd": [sys.executable, "-m", "uvicorn", "src.agents.rag_agent:app", "--port", "8105", "--host", "0.0.0.0"]},
    {"name": "Host Orchestrator Gradio UI (Google ADK)", "port": 8083, "cmd": [sys.executable, "-m", "src.agents.host_orchestrator"]},
    {"name": "Streamlit HITL Dashboard", "port": 8501, "cmd": [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.headless", "true"]}
]

running_processes: List[subprocess.Popen] = []

def cleanup(signum=None, frame=None):
    print("\n🛑 Shutting down all Agentic Healthcare AI microservices...")
    for proc in running_processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    print("✅ All services stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    print("=" * 75)
    print("🚀 BOOTING AGENTIC AI DISCHARGE SYSTEM (LANGGRAPH + ADK + AGNO + DUAL MCP)")
    print("=" * 75)

    for svc in SERVICES:
        print(f"▶ Starting {svc['name']:<48} [Port :{svc['port']}]")
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        p = subprocess.Popen(svc["cmd"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running_processes.append(p)
        time.sleep(0.6)

    print("\n⏳ Waiting for services to initialize...")
    time.sleep(3.5)

    print("\n" + "=" * 75)
    print("🩺 HEALTH CHECK MATRIX:")
    print("=" * 75)

    for svc in SERVICES:
        url = f"http://localhost:{svc['port']}/health" if svc['port'] != 8501 and svc['port'] != 8083 else f"http://localhost:{svc['port']}"
        status_str = "UNKNOWN"
        try:
            r = httpx.get(url, timeout=2.0)
            status_str = f"ONLINE (HTTP {r.status_code})" if r.status_code in [200, 302] else f"HTTP {r.status_code}"
        except Exception as e:
            status_str = "STARTING / OFFLINE"
        print(f"  {svc['name']:<45} Port :{svc['port']} -> {status_str}")

    print("\n" + "=" * 75)
    print("🌐 ACCESS POINTS:")
    print("  🖥️ Streamlit HITL Dashboard:  http://localhost:8501")
    print("  🎛️ Gradio Host Orchestrator:  http://localhost:8083")
    print("  📋 Primary MCP Server:        http://localhost:8200/clinicaltools/resources")
    print("  📊 Secondary MCP Server:      http://localhost:8201/analyticstools/tools")
    print("  🏥 Mock EHR API:              http://localhost:8050/docs")
    print("=" * 75)
    print("\nPress Ctrl+C to stop all services.\n")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
