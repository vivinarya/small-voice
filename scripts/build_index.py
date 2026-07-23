#!/usr/bin/env python
"""build_index.py — Build a FAISS retrieval index from any PDFs.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --src data/docs --out data/index --embed minilm

Drop any PDFs (academic, textbooks, manuals, reports, anything) under the
source directory.  No folder structure is required — all PDFs are found
recursively and indexed by document name + page number.

Prerequisites:
    pip install sentence-transformers faiss-cpu pypdf
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Force UTF-8 on stdout so log messages containing non-ASCII characters don't
# crash the logger on Windows consoles (cp1252). Mirrors src/main.py.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from retrieval.embedder import build_embedder
from retrieval.faiss_index import FAISSIndex
from retrieval.ingest import discover, ingest_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS retrieval index from any PDF documents."
    )
    parser.add_argument(
        "--src",
        default="data/docs",
        help="Source directory containing PDFs, searched recursively (default: data/docs)",
    )
    parser.add_argument(
        "--out",
        default="data/index",
        help="Output directory for index artifacts (default: data/index)",
    )
    parser.add_argument(
        "--embed",
        default="minilm",
        choices=["minilm", "bge_small"],
        help="Embedding backend (default: minilm)",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)

    logger.info("=" * 60)
    logger.info("FAISS Index Builder")
    logger.info("  Source  : %s", src_dir)
    logger.info("  Output  : %s", out_dir)
    logger.info("  Embedder: %s", args.embed)
    logger.info("=" * 60)

    if not os.path.exists(src_dir):
        logger.error("Source directory '%s' does not exist.", src_dir)
        logger.error("Create it and drop any PDFs inside (any folder depth).")
        sys.exit(1)

    # 1. Discover all PDFs recursively
    t0 = time.perf_counter()
    pdf_entries = discover(src_dir)
    if not pdf_entries:
        logger.warning("No PDFs found in '%s'. Index will be empty.", src_dir)
    else:
        logger.info("Found %d PDF file(s).", len(pdf_entries))

    # 2. Ingest into chunks
    all_chunks = []
    for pdf_path, source_name in pdf_entries:
        logger.info("Ingesting: %s", os.path.basename(pdf_path))
        chunks = ingest_pdf(pdf_path, source_name)
        all_chunks.extend(chunks)
        logger.info("  → %d chunks", len(chunks))

    logger.info("Total chunks: %d (%.1fs)", len(all_chunks), time.perf_counter() - t0)

    # 3. Load embedder
    logger.info("Loading embedder '%s'...", args.embed)
    embedder = build_embedder(args.embed)

    # 4. Build and persist FAISS index
    logger.info("Building FAISS index...")
    t1 = time.perf_counter()
    idx = FAISSIndex.build(all_chunks, embedder, out_dir)
    elapsed = time.perf_counter() - t1

    logger.info("=" * 60)
    logger.info("Index built successfully!")
    logger.info("  Vectors : %d", idx.ntotal)
    logger.info("  Elapsed : %.1fs", elapsed)
    logger.info("  Output  : %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
