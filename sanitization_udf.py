#!/usr/bin/env python3
"""
sanitize_threads_udf.py
────────────────────────────────────────────────────────────────────────────
PySpark script that:
  • Registers a UDF `sanitize_email_thread(raw: str) -> str`
  • Reads dirty_email_threads.csv (columns: sl.no, market, text)
  • Adds a `clean_text` column with sanitized content
  • Writes the cleaned result to emails_clean.csv

Sanitization logic removes HTML, RFC headers, disclaimers, ads, PII links,
mojibake, and extra whitespace while preserving salutations/closings.
"""

from __future__ import annotations
import re
import unicodedata
from html import unescape

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# ─────────────────────────── Regex helpers (compiled once) ──────────────────────────
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*?>.*?</\1>", re.I | re.S)
_HTML_TAG     = re.compile(r"<[^>]+>", re.S)

_HEADER_LINE = re.compile(
    r"^\s*(from|de|von|da|от|发件人|gönderen|寄件者|送信者|kimden|差出人|mittente|寄件人)\s*:\s*.+$",
    re.I,
)
_ORIGINAL_MSG_SEP = re.compile(
    r"^-{2,}\s*(original message|mensagem original|nachricht|ursprüngliche nachricht|"
    r"message d'origine|mensaje original|исходное письмо|原始邮件|元のメッセージ)\s*-{2,}$",
    re.I,
)
_ON_WROTE = re.compile(
    r"^\s*(on|am|el|le|il|на|于|於|日時|tarih|no dia)\b.*\b(wrote|schrieb|escribió|écrit|写道|yazdı)\s*:?\s*$",
    re.I,
)

_AUTOGEN = re.compile(
    r"(auto[- ]?generated|do not reply|no[- ]?reply|ne pas répondre|"
    r"nicht antworten|no responder|返信しないで|自動送信|自動通知|"
    r"automatic message|mensagem automática|mesaj otomatik|автоматическ)",
    re.I,
)
_DISCLAIMER = re.compile(
    r"(confidential|privileged|disclaimer|do not distribute|"
    r"no\s*divulgar|aviso legal|rechtlicher hinweis|haftung|"
    r"riservat[oa]|informazioni riservate|gizlilik|kvkk|unsubscribe|取消订阅|退订|"
    r"privacy|terms|conditions|条款|условия)",
    re.I,
)

_URL     = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
_EMAIL   = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_TEL_URI = re.compile(r"\btel:\+?\d[\d\-()\s]{5,}\b", re.I)
_PHONE   = re.compile(r"(?<!\w)(\+?\d[\d\-()\s]{6,}\d)(?!\w)")

_QUOTE_LINE = re.compile(r"^\s*>+.*$")
_MOJIBAKE   = re.compile(r"[�]{1,}|\uFFFD{1,}|(?:[�\uFFFD][\W_]{0,2}){2,}")

_MULTI_PUNCT = re.compile(r"([!?.,;:])\1{2,}")
_WS_LINES    = re.compile(r"[ \t\u00A0\u200B\u200C\u200D]{2,}")
_BLANKS      = re.compile(r"(?:\s*\n){3,}")

_SALUTATION = re.compile(
    r"^\s*(bonjour|hola|hallo|ciao|buongiorno|dear|hello|hi|"
    r"здравствуй|здравствуйте|merhaba|こんにちは|おはよう|你好|您好|namaste|नमस्ते)\b",
    re.I,
)
_CLOSING = re.compile(
    r"^\s*(regards|best|cheers|cordialement|saludos|mit freundlichen grüßen|"
    r"cordiali saluti|şimdiden teşekkürler|saygılar|ありがとうございます|此致|谢谢|धन्यवाद|thanks)\b",
    re.I,
)

# ───────────────────────────── Helper functions ────────────────────────────
def _normalize_unicode(text: str) -> str:
    text = unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00A0", " ").replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")

def _strip_html(text: str) -> str:
    return _HTML_TAG.sub(" ", _SCRIPT_STYLE.sub(" ", text))

def _drop_rfc_blocks(lines: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if _HEADER_LINE.match(ln) or _ORIGINAL_MSG_SEP.match(ln) or _ON_WROTE.match(ln) or _QUOTE_LINE.match(ln):
            i += 1
            while i < len(lines) and (
                _HEADER_LINE.match(lines[i]) or _QUOTE_LINE.match(lines[i]) or not lines[i].strip()
            ):
                i += 1
            continue
        out.append(ln)
        i += 1
    return out

def _drop_disclaimers_ads(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
        elif (_DISCLAIMER.search(s) or _AUTOGEN.search(s)) and not (_SALUTATION.match(s) or _CLOSING.match(s)):
            continue
        else:
            out.append(ln)
    return out

def _scrub_inline_piis(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _TEL_URI.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    return text

def _tidy(text: str) -> str:
    text = _MOJIBAKE.sub(" ", text)
    text = _MULTI_PUNCT.sub(r"\1", text)
    text = _WS_LINES.sub(" ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()

# ─────────────────────────── Core sanitization UDF ─────────────────────────
def sanitize_email_thread(raw: str) -> str:
    if not raw:
        return ""
    text = _scrub_inline_piis(_strip_html(_normalize_unicode(raw)))
    lines = text.splitlines()
    lines = _drop_rfc_blocks(lines)
    lines = _drop_disclaimers_ads(lines)
    return _tidy("\n".join(lines))

sanitize_email_thread_udf = udf(sanitize_email_thread, StringType())

# ────────────────────────────── Main job ───────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder.appName("email-thread-sanitize").getOrCreate()

    INPUT_PATH  = "dirty_email_threads.csv"  # raw data file
    OUTPUT_PATH = "emails_clean.csv"         # cleaned output
    DELIM       = ","                        # set to "\t" for TSV

    df = (
        spark.read
        .option("header", True)
        .option("sep", DELIM)
        .csv(INPUT_PATH)
    )

    text_col = {c.lower(): c for c in df.columns}.get("text", "text")

    df_clean = df.withColumn("clean_text", sanitize_email_thread_udf(col(text_col)))

    (
        df_clean
        .write
        .mode("overwrite")
        .option("header", True)
        .option("sep", DELIM)
        .csv(OUTPUT_PATH)
    )

    spark.stop()
