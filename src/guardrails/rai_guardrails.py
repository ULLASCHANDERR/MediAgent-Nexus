import re
from typing import Dict, Any, List, Tuple, Optional

class PIIRedactor:
    """Masks Personally Identifiable Information (PII) / Protected Health Information (PHI)."""
    
    # Patterns for Phone, Aadhaar (12 digits), PAN (5 letters, 4 numbers, 1 letter), SSN, Email
    PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    AADHAAR_PATTERN = re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b')
    PAN_PATTERN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

    @classmethod
    def redact_text(cls, text: str, known_names: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        if not text:
            return "", []
        redactions = []
        redacted = text

        # Redact Aadhaar
        if cls.AADHAAR_PATTERN.search(redacted):
            redacted = cls.AADHAAR_PATTERN.sub("[REDACTED_AADHAAR]", redacted)
            redactions.append("Aadhaar")

        # Redact PAN
        if cls.PAN_PATTERN.search(redacted):
            redacted = cls.PAN_PATTERN.sub("[REDACTED_PAN]", redacted)
            redactions.append("PAN")

        # Redact Phone
        if cls.PHONE_PATTERN.search(redacted):
            redacted = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
            redactions.append("Phone")

        # Redact Email
        if cls.EMAIL_PATTERN.search(redacted):
            redacted = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
            redactions.append("Email")

        # Redact known patient names if specified
        if known_names:
            for name in known_names:
                if len(name.strip()) > 2 and re.search(r'\b' + re.escape(name) + r'\b', redacted, re.IGNORECASE):
                    redacted = re.sub(r'\b' + re.escape(name) + r'\b', "[PATIENT_NAME]", redacted, flags=re.IGNORECASE)
                    redactions.append("PatientName")

        return redacted, list(set(redactions))


class HallucinationChecker:
    """Verifies that generated responses are strictly grounded in retrieved patient records."""

    @classmethod
    def check_faithfulness(cls, response: str, context_chunks: List[str]) -> Tuple[float, bool, str]:
        if not response or not context_chunks:
            return 0.0, False, "Empty response or context"
        
        # If response states information not available, it is 100% faithful to the grounding rule
        if "information is not available" in response.lower() or "i don’t know" in response.lower() or "i don't know" in response.lower():
            return 1.0, True, "Correct refusal based on grounding policy"

        full_context = " ".join(context_chunks).lower()
        response_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', response.lower()))
        
        # Exclude stop words
        stopwords = {"the", "and", "for", "with", "this", "that", "from", "patient", "was", "are", "were", "been", "have", "has", "had"}
        meaningful_words = [w for w in response_words if w not in stopwords]
        
        if not meaningful_words:
            return 1.0, True, "No clinical claims made"

        # Check what percentage of key terms appear in context
        supported = [w for w in meaningful_words if w in full_context]
        faithfulness_score = len(supported) / len(meaningful_words)

        is_passed = faithfulness_score >= 0.70
        reason = f"Faithfulness score {faithfulness_score:.2f} >= 0.70" if is_passed else f"Faithfulness score {faithfulness_score:.2f} < 0.70 (Potential Hallucination)"
        
        return round(faithfulness_score, 2), is_passed, reason


class PromptInjectionGuard:
    """Detects prompt injection and jailbreak attempts in user queries."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+the\s+above", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(an|the)?\s*unrestricted", re.IGNORECASE),
        re.compile(r"dan\s+mode", re.IGNORECASE),
        re.compile(r"system\s*prompt", re.IGNORECASE),
        re.compile(r"reveal\s+(your|the)\s+secret\s+key", re.IGNORECASE),
        re.compile(r"act\s+as\s+an\s+unethical", re.IGNORECASE),
        re.compile(r"drop\s+database", re.IGNORECASE),
        re.compile(r"<script.*?>", re.IGNORECASE),
    ]

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, str]:
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(query):
                return False, f"Prompt injection pattern detected: '{pattern.pattern}'"
        return True, "Query passed prompt injection security check."


class ToxicityFilter:
    """Filters offensive or toxic content from clinical instructions or inputs."""
    
    TOXIC_TERMS = [
        "hate", "stupid", "idiot", "kill", "harm", "lethal dose", "poison", "malpractice coverup"
    ]

    @classmethod
    def filter_text(cls, text: str) -> Tuple[bool, str, List[str]]:
        found_terms = []
        clean_text = text
        for term in cls.TOXIC_TERMS:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            if pattern.search(clean_text):
                found_terms.append(term)
                clean_text = pattern.sub("[REMOVED_TOXIC_TERM]", clean_text)
                
        is_clean = len(found_terms) == 0
        return is_clean, clean_text, found_terms


class GuardrailManager:
    """Central orchestrator for Responsible AI (RAI) guardrails and HITL escalation."""

    def __init__(self):
        self.pii_redactor = PIIRedactor()
        self.hallucination_checker = HallucinationChecker()
        self.injection_guard = PromptInjectionGuard()
        self.toxicity_filter = ToxicityFilter()

    def evaluate_discharge_case(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates whether HITL Escalation is required."""
        risk_level = validation_result.get("risk_level", "Low")
        discharge_blocked = validation_result.get("discharge_blocked", False)
        critical_issues = [
            issue for issue in validation_result.get("issues", [])
            if issue.get("severity") == "Critical"
        ]

        escalation_required = (risk_level == "High") or discharge_blocked or len(critical_issues) > 0

        action = "Escalate to Human-in-the-Loop Reviewer (MANDATORY)" if escalation_required else "Auto-Approve Permitted"

        return {
            "escalation_required": escalation_required,
            "action": action,
            "risk_level": risk_level,
            "discharge_blocked": discharge_blocked,
            "critical_issues_count": len(critical_issues),
            "reasons": [issue.get("description") for issue in critical_issues]
        }
