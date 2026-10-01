# Contributing to MediAgent-Nexus

Thank you for your interest in contributing to **MediAgent-Nexus**! We welcome contributions to enhance clinical AI workflows, improve Responsible AI guardrails, or add additional healthcare integrations.

## 🛠️ Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/ULLASCHANDERR/MediAgent-Nexus.git
   cd MediAgent-Nexus
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite**:
   ```bash
   python3 test_system.py
   ```

## 📋 Code Guidelines

- **Architecture Integrity**: Keep agent roles cleanly separated across their respective frameworks (LangGraph, Google ADK, Agno).
- **MCP Compliance**: When exposing or consuming tools, follow the 6 Model Context Protocol (MCP) primitives (Tools, Resources, Prompts, Sampling, Elicitation, Roots).
- **A2A Protocol**: Ensure all agents expose their `AgentCard` at `GET /.well-known/agent.json` and validate the `X-Agent-Auth-Token` header.
- **Safety First**: Never bypass or weaken Responsible AI guardrails (PII redaction, hallucination checks, prompt injection defense).

## 🚀 Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m "feat: add descriptive commit message"`
3. Push to your branch: `git push origin feature/your-feature-name`
4. Open a Pull Request on GitHub.
