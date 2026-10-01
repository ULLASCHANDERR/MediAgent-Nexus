import httpx
import json
from typing import Dict, Any, AsyncGenerator, Optional
from src.a2a_common.protocol import AUTH_HEADER_NAME, SHARED_SECRET, AgentCard, A2AMessage

class A2AClient:
    """Agent-to-Agent (A2A) Client supporting discovery, streaming, and non-streaming invocation."""
    
    def __init__(self, auth_token: str = SHARED_SECRET, timeout: float = 60.0):
        self.auth_token = auth_token
        self.timeout = timeout
        self.headers = {
            AUTH_HEADER_NAME: self.auth_token,
            "Content-Type": "application/json"
        }

    async def get_agent_card(self, base_url: str) -> Optional[AgentCard]:
        url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self.headers)
            if resp.status_code == 200:
                return AgentCard(**resp.json())
            return None

    async def send_message(self, agent_url: str, message: A2AMessage) -> Dict[str, Any]:
        """Invoke an A2A agent synchronously / non-streaming."""
        endpoint = f"{agent_url.rstrip('/')}/a2a/message"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(endpoint, json=message.model_dump(), headers=self.headers)
            if resp.status_code != 200:
                raise RuntimeError(f"A2A invocation failed on {endpoint} with status {resp.status_code}: {resp.text}")
            return resp.json()

    async def send_message_streaming(self, agent_url: str, message: A2AMessage) -> AsyncGenerator[Dict[str, Any], None]:
        """Invoke an A2A agent with streaming SSE events."""
        endpoint = f"{agent_url.rstrip('/')}/a2a/message/stream"
        msg_dict = message.model_dump()
        msg_dict["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", endpoint, json=msg_dict, headers=self.headers) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Streaming A2A call failed: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except Exception:
                            yield {"text": data_str}
