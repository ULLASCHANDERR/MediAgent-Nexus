import time
import uuid
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("Observability")

class LangfuseTelemetry:
    """Manages telemetry, traces, spans, and generation events with Langfuse & local audit trails."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LangfuseTelemetry, cls).__new__(cls)
            cls._instance.traces: Dict[str, Dict[str, Any]] = {}
            cls._instance.active_spans: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def create_trace(self, patient_id: str, trace_id: Optional[str] = None) -> str:
        tid = trace_id or f"trace-{patient_id}-{uuid.uuid4().hex[:8]}"
        self.traces[tid] = {
            "trace_id": tid,
            "patient_id": patient_id,
            "created_at": datetime.now().isoformat(),
            "spans": [],
            "generations": [],
            "sampling_events": [],
            "elicitation_events": [],
            "guardrail_events": [],
            "errors": [],
            "metadata": {}
        }
        return tid

    def start_span(self, trace_id: str, name: str, span_type: str = "agent_span", input_data: Any = None) -> str:
        span_id = f"span-{uuid.uuid4().hex[:8]}"
        span_record = {
            "span_id": span_id,
            "trace_id": trace_id,
            "name": name,
            "span_type": span_type,
            "start_time": time.time(),
            "start_iso": datetime.now().isoformat(),
            "input": input_data,
            "status": "RUNNING"
        }
        self.active_spans[span_id] = span_record
        return span_id

    def end_span(self, span_id: str, output_data: Any = None, status: str = "SUCCESS") -> Dict[str, Any]:
        span_record = self.active_spans.pop(span_id, None)
        if not span_record:
            return {}
        
        end_time = time.time()
        span_record["end_time"] = end_time
        span_record["duration_ms"] = round((end_time - span_record["start_time"]) * 1000, 2)
        span_record["output"] = output_data
        span_record["status"] = status

        tid = span_record["trace_id"]
        if tid in self.traces:
            self.traces[tid]["spans"].append(span_record)
        return span_record

    def log_llm_generation(self, trace_id: str, model_name: str, prompt: str, response: str, 
                           tokens_in: int = 150, tokens_out: int = 250, cost_est: float = 0.0004):
        generation_event = {
            "event_id": f"gen-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt": prompt[:300] + ("..." if len(prompt) > 300 else ""),
            "response": response[:300] + ("..." if len(response) > 300 else ""),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "estimated_cost_usd": cost_est
        }
        if trace_id in self.traces:
            self.traces[trace_id]["generations"].append(generation_event)
        return generation_event

    def log_sampling_event(self, trace_id: str, server_model_hints: List[str], client_selected_model: str, 
                           source_text: str, result_text: str, confidence: float):
        event = {
            "event_id": f"samp-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "server_hints": server_model_hints,
            "client_model": client_selected_model,
            "source_text": source_text[:200],
            "result_text": result_text[:200],
            "confidence": confidence
        }
        if trace_id in self.traces:
            self.traces[trace_id]["sampling_events"].append(event)
        return event

    def log_elicitation_event(self, trace_id: str, schema_sent: Dict[str, Any], action: str, reviewer_response: Any):
        event = {
            "event_id": f"elicit-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "schema_sent": schema_sent,
            "action": action, # accept, decline, cancel
            "reviewer_response": reviewer_response
        }
        if trace_id in self.traces:
            self.traces[trace_id]["elicitation_events"].append(event)
        return event

    def log_guardrail_event(self, trace_id: str, guardrail_name: str, check_result: str, 
                            action_taken: str, blocked: bool, details: Any = None):
        event = {
            "event_id": f"rai-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "guardrail_name": guardrail_name,
            "check_result": check_result,
            "action_taken": action_taken,
            "blocked": blocked,
            "details": details
        }
        if trace_id in self.traces:
            self.traces[trace_id]["guardrail_events"].append(event)
        return event

    def log_error_event(self, trace_id: str, exc_type: str, message: str, fallback_action: str):
        event = {
            "event_id": f"err-{uuid.uuid4().hex[:6]}",
            "timestamp": datetime.now().isoformat(),
            "exception_type": exc_type,
            "message": message,
            "fallback_action": fallback_action
        }
        if trace_id in self.traces:
            self.traces[trace_id]["errors"].append(event)
        return event

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self.traces.get(trace_id)

    def list_all_traces(self) -> List[Dict[str, Any]]:
        return list(self.traces.values())

tracker = LangfuseTelemetry()
