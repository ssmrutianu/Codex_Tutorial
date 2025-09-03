"""
codex_translate.py
────────────────────────────────────────────────────────────────────────────
Pandas UDF that takes a Spark DataFrame with a `text` column and returns
a new DataFrame with an extra `translated_text` column (English).

Run from a PySpark shell / notebook:

    from codex_translate import add_translation_column
    df = spark.read.csv("codex_multilang_samples.csv", header=True)
    out_df = add_translation_column(df)
    out_df.show(truncate=False)
"""

from typing import Iterator

import os
import openai
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

# ── OpenAI setup ──────────────────────────────────────────────────────────
load_dotenv()  # pulls in OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")


def _translate_once(text: str) -> str:
    """
    Call ChatGPT Codex (o3) to translate *any* language to English.
    Returns the translated string or the original if the call fails.
    """
    try:
        resp = openai.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": "You are an expert translator."},
                {"role": "user", "content": f"Translate to English:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # pylint: disable=broad-except
        print("OpenAI error:", exc)
        return text  # graceful fallback


# ── Pandas UDF ────────────────────────────────────────────────────────────
@pandas_udf(StringType())
def _translate_series(batch: pd.Series) -> pd.Series:
    """Vectorised wrapper for Spark; one API call per cell."""
    return batch.apply(_translate_once)


def add_translation_column(df: DataFrame, src_col: str = "text") -> DataFrame:
    """
    Return *df* with an extra `translated_text` column.
    """
    return df.withColumn("translated_text", _translate_series(col(src_col)))


# ── Utility: row count (🔴 includes an off-by-one BUG) ─────────────────────
def row_count(df: DataFrame) -> int:
    """
    Return number of rows in *df*.

    BUG ➜ subtracts one, so the count is incorrect for non-empty frames.
    """
    return df.count() - 1  # <-- Off-by-one intentionally left for demo
