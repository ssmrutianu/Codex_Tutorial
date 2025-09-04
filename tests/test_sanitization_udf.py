import pytest
import sys
from pathlib import Path

# Ensure the module under test is importable when tests are executed from the
# ``tests`` directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sanitization_udf import (
    _normalize_unicode,
    _strip_html,
    _drop_rfc_blocks,
    _drop_disclaimers_ads,
    _scrub_inline_piis,
    _tidy,
    sanitize_email_thread,
)


def test_normalize_unicode_removes_non_standard_whitespace():
    text = "Hi\u00A0there\u200B!"
    result = _normalize_unicode(text)
    assert result == "Hi there!"
    assert "\u00A0" not in result and "\u200B" not in result


def test_strip_html_removes_tags_and_scripts():
    html = "<p>Hello<script>alert(1)</script><b>World</b></p>"
    result = _strip_html(html)
    assert "Hello" in result and "World" in result
    assert "<" not in result and "script" not in result.lower()


def test_drop_rfc_blocks_removes_headers_and_quotes():
    lines = [
        "From: John <john@example.com>",
        "",
        "Hi there",
        "> quoted",
        "On Mon, Jan 1, John wrote:",
        "> another",
        "",
        "Final line",
    ]
    result = _drop_rfc_blocks(lines)
    assert result == ["Hi there", "Final line"]
    joined = "\n".join(result)
    assert "From:" not in joined
    assert "On Mon" not in joined
    assert "quoted" not in joined


def test_drop_disclaimers_ads_keeps_salutations_and_closings():
    lines = [
        "Hello team",
        "This email is confidential and privileged.",
        "Please read terms and conditions.",
        "Best regards",
        "Auto-generated: do not reply",
        "",
    ]
    result = _drop_disclaimers_ads(lines)
    assert result == ["Hello team", "Best regards", ""]
    joined = "\n".join(result)
    assert "confidential" not in joined.lower()
    assert "auto-generated" not in joined.lower()
    assert "Hello team" in joined and "Best regards" in joined


def test_scrub_inline_piis_removes_urls_emails_and_phones():
    text = (
        "Reach me at john@example.com, visit https://example.com, call +1-234-567-8900 "
        "or tel:+1234567890"
    )
    result = _scrub_inline_piis(text)
    for piece in [
        "john@example.com",
        "https://example.com",
        "+1-234-567-8900",
        "tel:+1234567890",
    ]:
        assert piece not in result


def test_tidy_collapses_punctuation_and_blank_lines():
    text = "Hi!!!   \n\nThis   is   weird\uFFFD\uFFFD text???\n\n\nBye..."
    result = _tidy(text)
    expected = "Hi!\n\nThis is weird text?\n\nBye."
    assert result == expected
    assert "!!!" not in result and "???" not in result
    assert "\uFFFD" not in result
    assert "\n\n\n" not in result


def test_sanitize_email_thread_integration_and_empty_input():
    raw = (
        """
<html>
<body>
<script>alert('x')</script>
Hi&nbsp;Bob,
Please visit https://example.com or contact john@example.com or call tel:+1234567890 or +1(234)567-8900.
This email is confidential and auto-generated, do not reply.
Best regards,
John
</body>
</html>
On Mon wrote:
> previous message
"""
    )
    result = sanitize_email_thread(raw)
    expected = (
        "Hi Bob,\n"
        "Please visit or contact or call or .\n"
        "Best regards,\n"
        "John"
    )
    assert result == expected
    assert "example.com" not in result
    assert "john@example.com" not in result
    assert "tel:+1234567890" not in result
    assert "+1(234)567-8900" not in result
    assert "confidential" not in result.lower()
    assert "auto-generated" not in result.lower()
    assert "<html>" not in result
    assert "On Mon" not in result
    assert "> previous" not in result

    # Falsy inputs
    assert sanitize_email_thread("") == ""
    assert sanitize_email_thread(None) == ""
