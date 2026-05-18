from app.pii import redact_pii


def test_email_redacted():
    result = redact_pii("Contact us at test@example.com for info.")
    assert "[EMAIL_REDACTED]" in result.redacted_text
    assert "test@example.com" not in result.redacted_text
    assert "email" in result.types_detected


def test_phone_redacted():
    result = redact_pii("Call us on 020 7946 0958 today.")
    assert "[PHONE_REDACTED]" in result.redacted_text
    assert "020 7946 0958" not in result.redacted_text
    assert "phone" in result.types_detected


def test_sort_code_redacted():
    result = redact_pii("Sort code: 12-34-56")
    assert "[SORT_CODE_REDACTED]" in result.redacted_text
    assert "12-34-56" not in result.redacted_text
    assert "sort_code" in result.types_detected


def test_pii_types_returned():
    text = "Email: a@b.com, Phone: 020 7946 0958, Sort: 12-34-56"
    result = redact_pii(text)
    assert "email" in result.types_detected
    assert "phone" in result.types_detected
    assert "sort_code" in result.types_detected
    assert result.redactions_applied == 3


def test_log_only_mode():
    text = "Email: a@b.com"
    result = redact_pii(text, pii_mode="log_only")
    assert "a@b.com" in result.redacted_text
    assert result.redactions_applied == 0
    assert "email" in result.types_detected
