"""
auto_metadata.py — LLM-Based Schema Extraction
================================================
Analyzes uploaded datasets to automatically detect metadata schema.
Returns the exact format that MetadataFilter / query_constructor expects.
"""

import json
import logging
from typing import List, Tuple

import pandas as pd
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Type mapping: AttributeInfo.type → Atlas Vector Search filter field type
# ──────────────────────────────────────────────────────────────────────────────
_ATLAS_TYPE_MAP = {
    "string": "token",
    "integer": "number",
    "float": "number",
    "[string]": "token",
}


def atlas_type_from_attribute_type(attr_type: str) -> str:
    """Map an AttributeInfo type string to an Atlas Vector Search index field type.

    Atlas supports: 'token' (filterable string/list), 'number' (numeric).
    """
    return _ATLAS_TYPE_MAP.get(attr_type, "token")


# ──────────────────────────────────────────────────────────────────────────────
# Schema extraction prompt
# ──────────────────────────────────────────────────────────────────────────────
_SCHEMA_EXTRACTION_PROMPT = """\
You are a data analyst. Given the following sample data (first rows of a dataset \
as JSON), analyze the schema and determine:

1. **content_column**: The column that contains the main text/content that \
describes each record. This is the column that would be used for semantic search. \
Pick the column with the longest, most descriptive text.

2. **content_description**: A brief one-sentence description of what the content \
column represents (e.g., "Brief summary of a movie", "Product description").

3. **metadata_fields**: For EVERY other column (excluding the content column), \
provide:
   - **name**: The exact column name
   - **description**: A human-readable description of what this field contains. \
If the field has a small set of distinct values, list them like: \
"Keywords for filtering: ['value1', 'value2', ...]"
   - **type**: One of exactly these strings:
     - "string" — for text, dates, categorical single-value fields
     - "integer" — for whole numbers
     - "float" — for decimal numbers
     - "[string]" — for fields that contain a list/array of strings

IMPORTANT RULES:
- Return ONLY valid JSON, no markdown, no explanation
- The content_column must be exactly one column name from the data
- metadata_fields must cover ALL remaining columns
- For date-like columns, use type "string"
- For columns with comma-separated tags/categories, use type "[string]"
- If a column has a small set of unique values (< 20), include those values in \
the description like: "Keywords for filtering: ['val1', 'val2', ...]"

Sample Data:
{sample_json}

Column Names: {columns}

Return JSON in exactly this format:
{{
  "content_column": "column_name",
  "content_description": "Brief description of the content",
  "metadata_fields": [
    {{"name": "col_name", "description": "Human description", "type": "string"}}
  ]
}}
"""


def extract_metadata_schema(
    df: pd.DataFrame,
    llm: BaseChatModel,
) -> Tuple[List[AttributeInfo], str, str]:
    """Analyze a DataFrame using an LLM and extract the metadata schema.

    Args:
        df: The uploaded DataFrame to analyze.
        llm: The configured LLM to use for schema analysis.

    Returns:
        Tuple of:
            - metadata_field_info: List[AttributeInfo] matching query_constructor format
            - document_content_description: str describing the content column
            - content_column: str name of the identified content column
    """
    # ── Prepare sample data for the LLM ──────────────────────────────────
    sample_size = min(10, len(df))
    sample_df = df.head(sample_size)

    # For columns with lists stored as strings, try to show them properly
    sample_json = sample_df.to_json(orient="records", indent=2, default_handler=str)

    # Collect unique values for categorical columns (helps LLM describe them)
    columns_info = []
    for col in df.columns:
        nunique = df[col].nunique()
        columns_info.append(f"{col} ({nunique} unique values)")

    prompt = _SCHEMA_EXTRACTION_PROMPT.format(
        sample_json=sample_json,
        columns=", ".join(columns_info),
    )

    logger.info("Sending schema extraction prompt to LLM...")
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    # ── Parse LLM response ───────────────────────────────────────────────
    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines)

    try:
        schema = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw: {raw_text}")
        raise ValueError(
            f"LLM returned invalid JSON. Please try again.\n\nRaw response:\n{raw_text}"
        )

    # ── Validate and convert to AttributeInfo list ───────────────────────
    content_column = schema.get("content_column", "")
    content_description = schema.get("content_description", "Dataset content")
    raw_fields = schema.get("metadata_fields", [])

    if content_column not in df.columns:
        # Fallback: pick the column with the longest average text length
        text_lengths = {
            col: df[col].astype(str).str.len().mean()
            for col in df.columns
        }
        content_column = max(text_lengths, key=text_lengths.get)
        logger.warning(
            f"LLM-identified content column not found. Falling back to: {content_column}"
        )

    valid_types = {"string", "integer", "float", "[string]"}
    metadata_field_info = []
    for field in raw_fields:
        name = field.get("name", "")
        if name == content_column or name not in df.columns:
            continue
        field_type = field.get("type", "string")
        if field_type not in valid_types:
            field_type = "string"
        metadata_field_info.append(
            AttributeInfo(
                name=name,
                description=field.get("description", f"The {name} field"),
                type=field_type,
            )
        )

    # Ensure we didn't miss any columns (safety net)
    covered_names = {f.name for f in metadata_field_info}
    for col in df.columns:
        if col != content_column and col not in covered_names:
            metadata_field_info.append(
                AttributeInfo(
                    name=col,
                    description=f"The {col} field",
                    type="string",
                )
            )

    logger.info(
        f"Schema extracted — content: '{content_column}', "
        f"metadata fields: {[f.name for f in metadata_field_info]}"
    )

    return metadata_field_info, content_description, content_column
