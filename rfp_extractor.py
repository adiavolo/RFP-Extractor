# rfp_extractor.py
# FINAL PRODUCTION VERSION
# - Clean, maintainable code
# - All current fixes included
# - No over-engineering
# - Robust error handling

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Project imports
from schema_and_prompts import (
    AssignmentSchema,
    ASSIGNMENT_KEYS,
    FIELD_QUERIES,
    K_MAP,
    build_single_pass_prompt,
    validate_extraction,
)

# =========================
# Environment & Logging
# =========================
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            "logs/rfp_extractor.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rfp-extractor")

# Quiet chatty libraries
for _name in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# API Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY missing. Set it in .env file.")
    sys.exit(2)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Constants
# =========================
DOC_BASE = "BASE_PDF"
DOC_ADDENDUM = "ADDENDUM"
DOC_HTML = "HTML"

HARD_FIELDS = {"Due Date", "Delivery Date",
               "Bid Submission Type", "Term of Bid"}
LIST_FIELDS = {"Any Additional Documentation Required",
               "Contract or Cooperative to use"}
ARRAY_FIELDS = {"Product", "Model_no", "Part_no", "Product Specification"}


@dataclass
class Chunk:
    """Document chunk with metadata"""
    text: str
    source_file: str
    doc_type: str
    page: Optional[int]
    chunk_id: str


# =========================
# Document Parsing
# =========================
def parse_pdf_to_chunks(
    pdf_path: str,
    chunk_size: int = 600,
    overlap: int = 100
) -> List[Chunk]:
    """
    Parse PDF into overlapping text chunks.
    Auto-classify as BASE_PDF or ADDENDUM based on filename patterns.
    """
    base = os.path.basename(pdf_path)

    # Determine document type
    doc_type = DOC_BASE
    lower = base.lower()
    if any(kw in lower for kw in ["addendum", "amendment", "clarification", "update"]):
        doc_type = DOC_ADDENDUM
        logger.info(f"Classified '{base}' as ADDENDUM")
    else:
        logger.info(f"Classified '{base}' as BASE_PDF")

    chunks = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text("text")

            # Clean text
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r'[\u2018\u2019]', "'", text)
            text = re.sub(r'[\u201C\u201D]', '"', text)
            text = re.sub(r'\s+', ' ', text).strip()

            if not text:
                continue

            # Create overlapping chunks
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk_id = f"{base}_p{page_num+1}_c{chunk_idx}"
                    chunks.append(Chunk(
                        text=chunk_text,
                        source_file=base,
                        doc_type=doc_type,
                        page=page_num + 1,
                        chunk_id=chunk_id
                    ))
                    chunk_idx += 1

                start = end - overlap
                if start >= len(text):
                    break

        logger.info(
            f"Parsed {base}: {len(chunks)} chunks from {num_pages} pages")

    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")
    finally:
        if doc is not None:
            try:
                doc.close()
            except:
                pass

    return chunks


def parse_html_to_chunks(
    html_path: str,
    chunk_size: int = 600,
    overlap: int = 100
) -> List[Chunk]:
    """Parse HTML file into chunks"""
    base = os.path.basename(html_path)
    chunks = []

    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script/style tags
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)

        # Clean text
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201C\u201D]', '"', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Create chunks
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{base}_c{chunk_idx}"
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=base,
                    doc_type=DOC_HTML,
                    page=None,
                    chunk_id=chunk_id
                ))
                chunk_idx += 1

            start = end - overlap
            if start >= len(text):
                break

        logger.info(f"Parsed {base}: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Error parsing {html_path}: {e}")
        return []

    return chunks


# =========================
# Embedding & Retrieval
# =========================
def get_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Get embeddings from OpenAI with batching"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Embedding batch {i//batch_size} failed: {e}")
            # Return zero vectors for failed batch
            all_embeddings.extend([[0.0] * 1536] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and chunks"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def classify_field_complexity(field_name: str) -> str:
    """Classify field complexity for K value selection"""
    if field_name in {"Product Specification", "Any Additional Documentation Required"}:
        return "complex"
    elif field_name in {"Bid Summary", "Payment Terms", "Contract or Cooperative to use"}:
        return "medium"
    else:
        return "simple"


def retrieve_for_field(
    field_name: str,
    all_chunks: List[Chunk],
    chunk_embeddings: np.ndarray,
    k_map: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K chunks for a field using multi-query retrieval.
    Returns list of dicts with text_with_label, source_file, page, doc_type, score.
    """
    queries = FIELD_QUERIES.get(field_name, [field_name])
    complexity = classify_field_complexity(field_name)
    k = k_map.get(complexity, 5)

    # Get query embeddings
    query_embeddings = get_embeddings(queries)

    # Compute similarities
    similarities = cosine_similarity(query_embeddings, chunk_embeddings)
    max_scores = similarities.max(axis=0)

    # Get top K indices
    top_indices = np.argsort(max_scores)[-k:][::-1]

    results = []
    for idx in top_indices:
        chunk = all_chunks[idx]
        score = float(max_scores[idx])

        # Format with source label
        page_str = f"page {chunk.page}" if chunk.page else "HTML"
        text_with_label = f"[{chunk.source_file}, {page_str}]\n{chunk.text}"

        results.append({
            "text_with_label": text_with_label,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "doc_type": chunk.doc_type,
            "score": score
        })

    return results


# =========================
# LLM Extraction
# =========================
def call_llm_for_extraction(prompt: str, debug_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Call LLM with structured output for extraction.
    Save debug prompt if path provided.
    """
    if debug_path:
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        logger.info(f"Debug prompt saved to {debug_path}")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured data from procurement documents. Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=4000
        )

        content = response.choices[0].message.content.strip()

        # Clean markdown fences if present
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

        # Parse JSON
        data = json.loads(content)
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"LLM response: {content[:500]}")
        return {}
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}


# =========================
# Normalization
# =========================
def normalize_date(date_str: Any) -> Optional[str]:
    """
    Normalize date strings to YYYY-MM-DD format.
    Strip hallucinated times.
    """
    if not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    if not date_str:
        return None

    # If already YYYY-MM-DD format (with or without time)
    match = re.match(r'(\d{4}-\d{2}-\d{2})', date_str)
    if match:
        return match.group(1)

    # MM/DD/YYYY format
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # If descriptive (e.g., "within 45 days"), keep as-is
    if any(word in date_str.lower() for word in ["within", "after", "upon", "days", "weeks"]):
        return date_str

    # Default: return cleaned string
    return date_str


def _coerce_string_list(value: Any) -> Optional[List[str]]:
    """Coerce value to list of non-empty strings"""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None

    result = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, (int, float)):
            result.append(str(item))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique if unique else None


def normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize LLM output:
    - Clean Bid Number (remove # prefix)
    - Normalize dates
    - Clean arrays and lists
    - Normalize contact info
    - Validate with Pydantic
    """
    raw = dict(raw)

    # Clean Bid Number
    if isinstance(raw.get("Bid Number"), str):
        raw["Bid Number"] = raw["Bid Number"].lstrip('#').strip() or None

    # Normalize dates
    for date_field in ("Due Date", "Delivery Date"):
        if date_field in raw:
            raw[date_field] = normalize_date(raw.get(date_field))

    # Clean string arrays
    for fld in ("Product", "Model_no", "Part_no"):
        raw[fld] = _coerce_string_list(raw.get(fld))

    # Clean list fields
    for fld in ("Any Additional Documentation Required", "Contract or Cooperative to use"):
        raw[fld] = _coerce_string_list(raw.get(fld))

    # Normalize contact_info
    ci = raw.get("contact_info")
    if isinstance(ci, list):
        ci_list = _coerce_string_list(ci) or []
        raw["contact_info"] = ", ".join(ci_list) if ci_list else None
    elif isinstance(ci, dict):
        parts = []
        for k in ("name", "email", "phone", "address"):
            v = ci.get(k)
            if isinstance(v, (int, float)):
                v = str(v)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        raw["contact_info"] = ", ".join(parts) if parts else None
    elif isinstance(ci, str):
        raw["contact_info"] = ci.strip() or None
    else:
        raw["contact_info"] = None

    # Normalize phone and email in contact_info
    if isinstance(raw.get("contact_info"), str):
        ci = raw["contact_info"]

        # Phone: xxx-xxx-xxxx format
        phone_pattern = r'[\(\)\s\.\-]*(\d{3})[\)\s\.\-]*(\d{3})[\s\.\-]*(\d{4})'
        ci = re.sub(phone_pattern,
                    lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}", ci)

        # Email: lowercase
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ci = re.sub(email_pattern, lambda m: m.group(1).lower(), ci)

        # Fix common spacing issues
        ci = re.sub(r',(\S)', r', \1', ci)  # Add space after comma

        raw["contact_info"] = ci

    # Product Specification: ensure list of dicts
    ps = raw.get("Product Specification")
    if ps is not None and not isinstance(ps, list):
        ps = [ps] if isinstance(ps, dict) else []

    if isinstance(ps, list):
        cleaned_ps = []
        for item in ps:
            if isinstance(item, dict):
                # Clean specs dict - remove nulls if not needed
                specs = item.get("specs", {})
                if isinstance(specs, dict):
                    # For non-computer products, remove unnecessary tech fields
                    name = item.get("name", "").lower()
                    if "dock" in name or "accessory" in name or "peripheral" in name:
                        # Keep only relevant fields
                        cleaned_specs = {
                            k: v for k, v in specs.items()
                            if k in ["certification", "condition", "requirement"] and v is not None
                        }
                        item["specs"] = cleaned_specs

                cleaned_ps.append(item)
        raw["Product Specification"] = cleaned_ps if cleaned_ps else None

    # Validate with Pydantic
    try:
        model = AssignmentSchema(**raw)
        validated = model.model_dump(by_alias=True, exclude_none=False)
    except ValidationError as e:
        logger.warning(f"Pydantic validation errors: {e}")
        validated = raw

    # Return in assignment order
    ordered = {}
    for key in ASSIGNMENT_KEYS:
        if key in validated:
            ordered[key] = validated[key]

    return ordered


# =========================
# Provenance
# =========================
def build_provenance(
    final_data: Dict[str, Any],
    snippets: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build provenance for each field.
    Uses highest-precedence source from retrieved snippets.
    """
    prov: Dict[str, List[Dict[str, Any]]] = {}
    precedence = [DOC_ADDENDUM, DOC_BASE, DOC_HTML]

    for field, value in final_data.items():
        if value in (None, [], ""):
            continue

        snips = snippets.get(field, [])
        if not snips:
            continue

        # Find highest precedence source
        chosen = None
        for dt in precedence:
            chosen = next((s for s in snips if s["doc_type"] == dt), None)
            if chosen:
                break

        if not chosen:
            chosen = snips[0]

        # Calculate confidence score
        confidence = chosen.get("score", 0.5)

        # Build provenance entry
        entry = {
            "value": value,
            "source": chosen["source_file"],
            "page": chosen["page"],
            "final": True,
            "confidence_score": round(confidence, 3)
        }

        # Add contributing sources for arrays/lists
        if isinstance(value, list) and len(snips) > 1:
            entry["contributing_sources"] = [
                {
                    "source": s["source_file"],
                    "page": s["page"],
                    "doc_type": s["doc_type"]
                }
                for s in snips[:3]  # Top 3 sources
            ]

        prov[field] = [entry]

    return prov


# =========================
# Main Pipeline
# =========================
def extract_rfp_data(
    input_files: List[str],
    output_dir: str,
    bid_id: str,
    enable_provenance: bool = False,
    debug: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Main extraction pipeline.
    Returns: (output_json_path, provenance_json_path)
    """
    logger.info(f"Starting extraction for Bid ID: {bid_id}")
    logger.info(f"Input files: {input_files}")

    # 1. Parse all documents
    all_chunks: List[Chunk] = []
    for file_path in input_files:
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        if file_path.lower().endswith('.pdf'):
            chunks = parse_pdf_to_chunks(file_path)
        elif file_path.lower().endswith(('.html', '.htm')):
            chunks = parse_html_to_chunks(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            continue

        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks extracted from input files")
        return "", None

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Safety check: warn if very few chunks (likely PDF parsing failed)
    if len(all_chunks) < 20:
        logger.warning(
            f"⚠️  WARNING: Only {len(all_chunks)} chunks extracted. "
            "If you provided PDF files, they may have failed to parse. "
            "Check for 'Error parsing' messages above."
        )

    # 2. Embed chunks
    logger.info("Generating embeddings...")
    chunk_texts = [c.text for c in all_chunks]
    chunk_embeddings = get_embeddings(chunk_texts)
    logger.info("Embeddings complete")

    # 3. Retrieve for each field
    logger.info("Retrieving relevant snippets for each field...")
    all_snippets: Dict[str, List[Dict[str, Any]]] = {}

    for field in ASSIGNMENT_KEYS:
        snippets = retrieve_for_field(
            field, all_chunks, chunk_embeddings, K_MAP)
        all_snippets[field] = snippets
        logger.debug(f"{field}: Retrieved {len(snippets)} snippets")

    # 4. Build prompt
    prompt = build_single_pass_prompt(all_snippets)

    # 5. Call LLM
    logger.info("Calling LLM for extraction...")
    debug_path = os.path.join(
        output_dir, f"{bid_id}_debug_prompt.txt") if debug else None
    raw_data = call_llm_for_extraction(prompt, debug_path)

    if not raw_data:
        logger.error("LLM returned empty data")
        return "", None

    # 6. Normalize output
    logger.info("Normalizing output...")
    final_data = normalize_output(raw_data)

    # 7. Validate
    issues = validate_extraction(final_data)
    if issues.get("errors"):
        for error in issues["errors"]:
            logger.error(f"Validation error: {error}")
    if issues.get("warnings"):
        for warning in issues["warnings"]:
            logger.warning(f"Validation warning: {warning}")

    # 8. Write output
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{bid_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Output written to {output_path}")

    # 9. Provenance (optional)
    prov_path = None
    if enable_provenance:
        prov_data = build_provenance(final_data, all_snippets)
        prov_path = os.path.join(output_dir, f"{bid_id}_provenance.json")
        with open(prov_path, 'w', encoding='utf-8') as f:
            json.dump(prov_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Provenance written to {prov_path}")

    logger.info("Extraction complete!")
    return output_path, prov_path


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="RFP Data Extractor")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input PDF/HTML files"
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--bid_id",
        required=True,
        help="Bid identifier (e.g., E20P4600040)"
    )
    parser.add_argument(
        "--provenance",
        choices=["on", "off"],
        default="off",
        help="Enable provenance tracking (default: off)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug prompt to file"
    )

    args = parser.parse_args()

    enable_prov = (args.provenance == "on")

    try:
        output_path, prov_path = extract_rfp_data(
            input_files=args.inputs,
            output_dir=args.out,
            bid_id=args.bid_id,
            enable_provenance=enable_prov,
            debug=args.debug
        )

        if output_path:
            print(f"\n✅ Success!")
            print(f"Output: {output_path}")
            if prov_path:
                print(f"Provenance: {prov_path}")
        else:
            print("\n❌ Extraction failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
