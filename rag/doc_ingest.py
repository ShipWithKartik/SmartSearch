"""
doc_ingest.py — Unstructured Document Ingestion (PDF / DOCX)
=============================================================
A second, independent ingestion path that sits alongside the structured
CSV/JSON pipeline in ingest.py. That module is untouched: this one handles
free text.

Flow:  extract text (per page) -> recursive character chunking ->
       embed with the SAME configured embedding model -> MongoDB Atlas,
       with metadata {source_file, page_number, chunk_index}.

Note on scanned PDFs: extraction reads the PDF text layer only. Image-only
(scanned) pages yield no text; those pages are reported back as warnings
rather than silently ingested as empty chunks. OCR would need a separate
dependency and is intentionally out of scope here.
"""

import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config_loader import config
from rag.utils.mongodb_helper import get_mongo_collection, create_vector_search_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")

# Metadata fields the document collection filters on, in Atlas index terms.
DOCUMENT_FILTER_FIELDS: Dict[str, str] = {
    "source_file": "token",
    "page_number": "number",
    "chunk_index": "number",
}


def default_document_collection() -> str:
    """Collection used for unstructured chunks (never the structured one)."""
    return config.get("document_collection_name", "documents-store")


def default_chunk_size() -> int:
    return int(config.get("chunk_size", 1000))


def default_chunk_overlap() -> int:
    return int(config.get("chunk_overlap", 150))


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DocumentIngestResult:
    """Summary of an unstructured ingestion run, for the UI to render."""

    collection_name: str = ""
    files_ingested: List[str] = field(default_factory=list)
    chunks_inserted: int = 0
    pages_extracted: int = 0
    empty_pages: int = 0
    skipped: List[Dict] = field(default_factory=list)   # [{file, reason}]
    warnings: List[str] = field(default_factory=list)
    per_file: List[Dict] = field(default_factory=list)  # [{file, pages, chunks}]


# ──────────────────────────────────────────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────────────────────────────────────────
def _read_bytes(file_obj: Union[bytes, "os.PathLike", str, BytesIO]) -> bytes:
    """Accept raw bytes, a filesystem path, or any file-like object."""
    if isinstance(file_obj, bytes):
        return file_obj
    if isinstance(file_obj, (str, os.PathLike)):
        with open(file_obj, "rb") as fh:
            return fh.read()
    # Streamlit UploadedFile and BytesIO both satisfy this.
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    if hasattr(file_obj, "read"):
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return file_obj.read()
    raise TypeError(f"Cannot read bytes from object of type {type(file_obj)}")


def extract_pdf_pages(data: bytes) -> List[Tuple[int, str]]:
    """Extract text per page from a PDF. Returns [(page_number, text)], 1-indexed.

    Pages with no extractable text layer are returned with an empty string so
    the caller can count and report them.
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page {i}: {e}")
            text = ""
        pages.append((i, text.strip()))
    return pages


def extract_docx_pages(data: bytes) -> List[Tuple[int, str]]:
    """Extract text from a DOCX.

    DOCX has no fixed pagination (pages are a rendering artifact), so the whole
    document is returned as a single logical page 1. Paragraphs and table cells
    are both collected, in document order where possible.
    """
    import docx

    document = docx.Document(BytesIO(data))

    parts = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return [(1, "\n\n".join(parts).strip())]


def extract_pages(data: bytes, filename: str) -> List[Tuple[int, str]]:
    """Dispatch to the right extractor based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_pages(data)
    if lower.endswith(".docx"):
        return extract_docx_pages(data)
    raise ValueError(
        f"Unsupported file type: {filename}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────
def chunk_pages(
    pages: List[Tuple[int, str]],
    source_file: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    start_index: int = 0,
) -> List[Document]:
    """Split extracted pages into overlapping chunks with source metadata.

    chunk_index is sequential across the whole file (not reset per page), so a
    citation like "report.pdf p.4 #17" is unambiguous.
    """
    chunk_size = chunk_size or default_chunk_size()
    chunk_overlap = chunk_overlap or default_chunk_overlap()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: List[Document] = []
    index = start_index

    for page_number, text in pages:
        if not text or not text.strip():
            continue
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source_file": source_file,
                        "page_number": int(page_number),
                        "chunk_index": index,
                    },
                )
            )
            index += 1

    return documents


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────
def build_documents_from_files(
    files: List[Tuple[str, Union[bytes, BytesIO]]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    result: Optional[DocumentIngestResult] = None,
) -> List[Document]:
    """Extract + chunk a batch of (filename, file-like) pairs into Documents.

    Failures on one file never abort the batch: they are recorded in
    result.skipped and the remaining files still ingest.
    """
    result = result if result is not None else DocumentIngestResult()
    all_docs: List[Document] = []

    for filename, file_obj in files:
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            result.skipped.append({"file": filename, "reason": "Unsupported file type"})
            continue

        try:
            data = _read_bytes(file_obj)
            pages = extract_pages(data, filename)
        except Exception as e:
            logger.error(f"Extraction failed for {filename}: {e}")
            result.skipped.append({"file": filename, "reason": f"Extraction failed: {e}"})
            continue

        non_empty = [(n, t) for n, t in pages if t.strip()]
        empty_count = len(pages) - len(non_empty)
        result.pages_extracted += len(non_empty)
        result.empty_pages += empty_count

        if not non_empty:
            result.skipped.append(
                {
                    "file": filename,
                    "reason": "No extractable text — the file may be a scanned/image-only "
                              "PDF, which needs OCR (not supported).",
                }
            )
            continue

        if empty_count:
            result.warnings.append(
                f"{filename}: {empty_count} page(s) had no text layer and were skipped "
                f"(likely scanned images)."
            )

        # chunk_index continues across files so every chunk in the collection
        # has a globally unique (source_file, chunk_index) pair.
        docs = chunk_pages(
            non_empty,
            source_file=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=0,
        )
        if not docs:
            result.skipped.append({"file": filename, "reason": "Produced no chunks"})
            continue

        result.files_ingested.append(filename)
        result.per_file.append(
            {"file": filename, "pages": len(non_empty), "chunks": len(docs)}
        )
        all_docs.extend(docs)

    return all_docs


def ingest_documents(
    files: List[Tuple[str, Union[bytes, BytesIO]]],
    collection_name: Optional[str] = None,
    replace: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    progress_callback=None,
) -> DocumentIngestResult:
    """Ingest PDF/DOCX files into MongoDB Atlas as embedded chunks.

    Args:
        files: [(filename, bytes-or-file-like)] to ingest.
        collection_name: Target collection. Defaults to config
            'document_collection_name', kept separate from the structured one.
        replace: Clear the collection (and rebuild its index) before inserting.
            When False, chunks are appended to whatever is already there.
        chunk_size / chunk_overlap: Override the config defaults.
        progress_callback: Optional callable(step: str, progress: float).

    Returns:
        DocumentIngestResult summarising what was ingested and what was skipped.
    """

    def _report(step: str, progress: float):
        if progress_callback:
            progress_callback(step, progress)
        logger.info(f"[DocIngest] {step} ({progress:.0%})")

    collection_name = collection_name or default_document_collection()
    result = DocumentIngestResult(collection_name=collection_name)

    if not files:
        raise ValueError("No files supplied for ingestion.")

    # ── Step 1: Extract and chunk ────────────────────────────────────────
    _report("Extracting text from documents…", 0.15)
    documents = build_documents_from_files(
        files, chunk_size=chunk_size, chunk_overlap=chunk_overlap, result=result
    )

    if not documents:
        reasons = "; ".join(f"{s['file']}: {s['reason']}" for s in result.skipped)
        raise ValueError(f"No text could be extracted from the uploaded file(s). {reasons}")

    _report(f"Chunked into {len(documents)} pieces…", 0.35)

    # ── Step 2: Prepare collection ───────────────────────────────────────
    database_name = config["database_name"]
    collection = get_mongo_collection(db_name=database_name, collection_name=collection_name)

    if replace:
        _report("Clearing existing document chunks…", 0.4)
        collection.delete_many({})
        try:
            collection.drop_search_index(config.get("vector_index_name", "default"))
            logger.info("Dropped existing document search index")
        except Exception as e:
            logger.info(f"No existing search index to drop (or error): {e}")

    # ── Step 3: Embed and insert ─────────────────────────────────────────
    _report("Embedding & inserting chunks…", 0.55)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    MongoDBAtlasVectorSearch.from_documents(documents, embeddings, collection=collection)
    result.chunks_inserted = len(documents)
    logger.info(f"Inserted {len(documents)} chunks into '{collection_name}'")

    # ── Step 4: Vector search index ──────────────────────────────────────
    _report("Creating vector search index…", 0.85)
    try:
        create_vector_search_index(
            collection=collection,
            index_name=config.get("vector_index_name", "default"),
            embedded_field_names=["embedding"],
            dimensions=config.get("embedding_model_dimensions", 384),
            similarity=config.get("similarity", "cosine"),
            filter_fields_with_datatype=DOCUMENT_FILTER_FIELDS,
        )
        logger.info("Document vector search index created successfully")
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {e}")
        result.warnings.append(f"Index creation: {e}")

    _report("Document ingestion complete!", 1.0)
    return result


def list_ingested_sources(collection_name: Optional[str] = None) -> List[str]:
    """Distinct source_file values currently in the document collection."""
    collection_name = collection_name or default_document_collection()
    try:
        collection = get_mongo_collection(
            db_name=config["database_name"], collection_name=collection_name
        )
        return sorted(str(s) for s in collection.distinct("source_file") if s)
    except Exception as e:
        logger.warning(f"Could not list ingested sources: {e}")
        return []
