import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+44\s?\d[\d\s]{8,10}\d|0\d[\d\s]{8,10}\d)")
SORT_CODE_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")


@dataclass
class RedactionResult:
    redacted_text: str
    redactions_applied: int
    types_detected: list[str]


def redact_pii(text: str, pii_mode: str = "redact_before_llm") -> RedactionResult:
    types_detected = []
    redactions_applied = 0

    if EMAIL_RE.search(text):
        types_detected.append("email")
    if PHONE_RE.search(text):
        types_detected.append("phone")
    if SORT_CODE_RE.search(text):
        types_detected.append("sort_code")

    if pii_mode == "log_only":
        return RedactionResult(
            redacted_text=text,
            redactions_applied=0,
            types_detected=types_detected,
        )

    redacted = text
    email_matches = EMAIL_RE.findall(redacted)
    redactions_applied += len(email_matches)
    redacted = EMAIL_RE.sub("[EMAIL_REDACTED]", redacted)

    phone_matches = PHONE_RE.findall(redacted)
    redactions_applied += len(phone_matches)
    redacted = PHONE_RE.sub("[PHONE_REDACTED]", redacted)

    sort_matches = SORT_CODE_RE.findall(redacted)
    redactions_applied += len(sort_matches)
    redacted = SORT_CODE_RE.sub("[SORT_CODE_REDACTED]", redacted)

    return RedactionResult(
        redacted_text=redacted,
        redactions_applied=redactions_applied,
        types_detected=types_detected,
    )
