"""
ingest.py — Dynamic Dataset Ingestion Pipeline
================================================
Converts a DataFrame into embedded LangChain Documents and inserts them
into a MongoDB Atlas collection with a vector search index.
"""

import logging
from typing import Dict, List

import pandas as pd
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from rag.auto_metadata import atlas_type_from_attribute_type
from rag.config_loader import config
from rag.utils.mongodb_helper import get_mongo_collection, create_vector_search_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _coerce_metadata_value(value, field_type: str):
    """Coerce a raw DataFrame cell into the correct type for metadata storage.

    Handles common edge cases like:
    - NaN → None
    - Stringified lists → actual lists
    - Numeric strings → float/int
    """
    if pd.isna(value):
        return None

    if field_type == "[string]":
        if isinstance(value, list):
            return [str(v) for v in value]
        # Handle comma-separated strings like "action, comedy, drama"
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            # Try to parse as JSON list
            import json
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback: split by comma
        return [v.strip() for v in s.split(",") if v.strip()]

    if field_type == "integer":
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return str(value)

    if field_type == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)

    # Default: string
    return str(value)


def _build_documents(
    df: pd.DataFrame,
    content_column: str,
    metadata_field_info: List[AttributeInfo],
) -> List[Document]:
    """Convert a DataFrame into a list of LangChain Documents.

    Args:
        df: The source DataFrame.
        content_column: Column name to use as page_content.
        metadata_field_info: List of AttributeInfo defining metadata fields.

    Returns:
        List of Document objects ready for embedding.
    """
    field_type_map = {f.name: f.type for f in metadata_field_info}
    documents = []

    for _, row in df.iterrows():
        content = str(row.get(content_column, ""))
        if not content.strip():
            continue

        metadata = {}
        for field in metadata_field_info:
            if field.name in row.index:
                val = _coerce_metadata_value(row[field.name], field.type)
                if val is not None:
                    metadata[field.name] = val

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def ingest_dataset(
    df: pd.DataFrame,
    metadata_field_info: List[AttributeInfo],
    content_column: str,
    collection_name: str,
    progress_callback=None,
) -> int:
    """Ingest a DataFrame into MongoDB Atlas with vector embeddings and search index.

    Args:
        df: The source DataFrame.
        metadata_field_info: Schema describing metadata fields.
        content_column: Column name containing the main text content.
        collection_name: MongoDB collection to insert into.
        progress_callback: Optional callable(step: str, progress: float) for UI updates.

    Returns:
        Number of documents inserted.
    """

    def _report(step: str, progress: float):
        if progress_callback:
            progress_callback(step, progress)
        logger.info(f"[Ingest] {step} ({progress:.0%})")

    database_name = config["database_name"]

    # ── Step 1: Get/create collection and clear it ───────────────────────
    _report("Preparing collection…", 0.1)
    collection = get_mongo_collection(db_name=database_name, collection_name=collection_name)
    collection.delete_many({})
    logger.info(f"Cleared collection: {collection_name}")

    # ── Step 2: Drop existing search index (if any) ──────────────────────
    _report("Dropping old search index…", 0.2)
    try:
        collection.drop_search_index("default")
        logger.info("Dropped existing 'default' search index")
    except Exception as e:
        logger.info(f"No existing search index to drop (or error): {e}")

    # ── Step 3: Convert DataFrame to Documents ───────────────────────────
    _report("Converting data to documents…", 0.3)
    documents = _build_documents(df, content_column, metadata_field_info)
    if not documents:
        raise ValueError("No valid documents were created from the uploaded data.")
    logger.info(f"Created {len(documents)} documents")

    # ── Step 4: Build filter fields for Atlas index ──────────────────────
    filter_fields: Dict[str, str] = {}
    for field in metadata_field_info:
        filter_fields[field.name] = atlas_type_from_attribute_type(field.type)
    logger.info(f"Filter fields for index: {filter_fields}")

    # ── Step 5: Create embeddings and insert ─────────────────────────────
    _report("Embedding & inserting documents…", 0.5)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    MongoDBAtlasVectorSearch.from_documents(documents, embeddings, collection=collection)
    logger.info(f"Inserted {len(documents)} documents with embeddings")

    # ── Step 6: Recreate vector search index ─────────────────────────────
    _report("Creating vector search index…", 0.8)
    try:
        create_vector_search_index(
            collection=collection,
            index_name=config.get("vector_index_name", "default"),
            embedded_field_names=["embedding"],
            dimensions=config.get("embedding_model_dimensions", 384),
            similarity=config.get("similarity", "cosine"),
            filter_fields_with_datatype=filter_fields,
        )
        logger.info("Vector search index created successfully")
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {e}")

    _report("Ingestion complete!", 1.0)
    return len(documents)
