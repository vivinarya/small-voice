# src/retrieval/ingest.py
"""Generic PDF ingestion — any PDF, any topic, no folder conventions required.

Usage:
    Drop PDFs anywhere under the source directory (flat or nested).
    The document name is inferred from the filename.

Requires: pip install pypdf  (PyMuPDF/fitz used automatically if installed)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from .base import Chunk

logger = logging.getLogger(__name__)

TARGET_TOKENS = 300   # approximate tokens per chunk
OVERLAP_TOKENS = 50   # overlap between adjacent chunks


def _token_count(text: str) -> int:
    return len(text.split())


def _clean_text(text: str) -> str:
    """Normalize PDF-extracted text."""
    text = unicodedata.normalize("NFC", text)
    # De-hyphenate line-break hyphens
    text = re.sub(r'-\n(\w)', r'\1', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def _slug(text: str) -> str:
    """Filesystem-safe slug, max 20 chars."""
    slug = text.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '_', slug)
    return (slug[:20]).strip('_') or "doc"


def _sliding_window_chunks(
    text: str,
    target: int = TARGET_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping word-token chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(piece)
        if end == len(words):
            break
        start += max(1, target - overlap)
    return chunks


def _read_pages_pypdf(pdf_path: str) -> list[tuple[int, str]]:
    try:
        import pypdf  # noqa: PLC0415
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def _read_pages_fitz(pdf_path: str) -> list[tuple[int, str]]:
    import fitz  # noqa: PLC0415
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages


def _read_pages(pdf_path: str) -> list[tuple[int, str]]:
    try:
        return _read_pages_fitz(pdf_path)
    except ImportError:
        return _read_pages_pypdf(pdf_path)


def ingest_pdf(pdf_path: str, source_name: str | None = None) -> list[Chunk]:
    """Ingest any PDF into page-aware Chunk objects.

    Args:
        pdf_path:    Path to any PDF file.
        source_name: Human-readable document name shown in citations.
                     Defaults to the filename stem.

    Returns list of Chunk objects with page numbers and source attribution.
    """
    path = Path(pdf_path)
    if source_name is None:
        # Clean up filename for display: underscores/hyphens → spaces, title-case
        source_name = path.stem.replace("_", " ").replace("-", " ").title()

    doc_slug = _slug(path.stem)
    chunks: list[Chunk] = []

    try:
        pages = _read_pages(pdf_path)
    except Exception as exc:
        logger.error("Failed to read PDF '%s': %s", pdf_path, exc)
        return []

    if not pages:
        logger.warning("No extractable text in '%s' — is it a scanned image PDF?", pdf_path)
        return []

    for page_num, raw_text in pages:
        text = _clean_text(raw_text)
        if not text:
            continue
        for local_idx, piece in enumerate(_sliding_window_chunks(text)):
            cid = f"{doc_slug}_p{page_num}_{local_idx}"
            chunks.append(Chunk(
                id=cid,
                text=piece,
                source=source_name,
                page=page_num,
                # Legacy fields left at defaults (klass=0, subject="", chapter="")
            ))

    logger.info("Ingested %d chunks from '%s' (%d pages with text)",
                len(chunks), path.name, len(pages))
    return chunks


def discover(src_dir: str) -> list[tuple[str, str]]:
    """Discover all PDFs recursively under src_dir.

    Returns list of (pdf_path, source_name) tuples.
    Any PDF found at any depth is included — no class/subject folder
    structure is required or expected.
    """
    src = Path(src_dir)
    if not src.exists():
        logger.warning("Source directory '%s' does not exist.", src_dir)
        return []

    results = []
    for pdf_file in sorted(src.rglob("*.pdf")):
        source_name = pdf_file.stem.replace("_", " ").replace("-", " ").title()
        results.append((str(pdf_file), source_name))

    logger.info("Discovered %d PDF(s) in '%s'", len(results), src_dir)
    return results
