# rfp_extractor.py
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# ---- Project-local imports (schema, queries, prompt) ----
from schema_and_prompts import (
    AssignmentSchema,
    ASSIGNMENT_KEYS,
    FIELD_QUERIES,
    K_MAP,
    build_single_pass_prompt,
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

# Quiet very chatty libs
for _name in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.error(
        "OPENAI_API_KEY missing. Set it in your .env (OPENAI_API_KEY=sk-...).")
    sys.exit(2)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Constants & Types
# =========================
DOC_BASE = "BASE_PDF"
DOC_ADDENDUM = "ADDENDUM"
DOC_HTML = "HTML"

HARD_FIELDS = {"Due Date", "Delivery Date",
               "Bid Submission Type", "Term of Bid"}
LIST_FIELDS = {"Any Additional Documentation Required",
               "Contract or Cooperative to use"}
# Product Specification is array of objects
ARRAY_FIELDS = {"Product", "Model_no", "Part_no", "Product Specification"}


@dataclass
class Chunk:
    text: str
    source_file: str
    doc_type: str  # BASE_PDF | ADDENDUM | HTML
    page: Optional[int]  # 1-indexed for PDFs; None for HTML
    chunk_id: str        # internal-only, not used in provenance


# =========================
# Utilities
# =========================
def is_addendum_by_heuristics(filename: str, first_page_text: str) -> bool:
    """
    Filename heuristic + first 100 chars. Avoid scanning whole page to prevent false positives like
    'Any addendums will be posted...'
    """
    fn = filename.lower()
    if any(k in fn for k in ["addendum", "amendment", "add_"]):
        return True
    head = (first_page_text or "")[:100].upper()
    return ("ADDENDUM" in head) or ("AMENDMENT" in head)


def normalize_phone_us_like(s: str) -> str:
    sdigits = re.sub(r"\D+", "", s or "")
    if len(sdigits) == 10:
        return f"{sdigits[0:3]}-{sdigits[3:6]}-{sdigits[6:10]}"
    return s


# === Added helpers (dedupe + string-list coercion) ===
def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _coerce_string_list(val) -> Optional[List[str]]:
    """
    Accepts list/str/number/None and returns a clean list[str] or None.
    - Removes None/empty items
    - Trims whitespace
    - Converts numbers to strings
    - If input is a single comma/semicolon/line-break separated string, split it
    """
    if val is None:
        return None
    # If already a list, clean each element
    if isinstance(val, list):
        out: List[str] = []
        for item in val:
            if item is None:
                continue
            if isinstance(item, (int, float)):
                s = str(item)
            elif isinstance(item, str):
                s = item.strip()
            else:
                # if it's a dict/object (e.g., accidental), skip for string lists
                s = None
            if s:
                out.append(s)
        out = _dedupe_preserve_order(out)
        return out or None

    # If it's a string, split on common separators
    if isinstance(val, str):
        parts = [p.strip() for p in re.split(r"[,\n;]+", val) if p.strip()]
        return _dedupe_preserve_order(parts) or None

    # If number, wrap as one string
    if isinstance(val, (int, float)):
        return [str(val)]

    return None


# =========================
# Parsing (PDF, HTML)
# =========================
def parse_pdf_to_chunks(path: str, doc_type: str) -> List[Chunk]:
    """Page-based chunking. One chunk per page. Split into ~600-word segments if page is long."""
    chunks: List[Chunk] = []
    base = os.path.basename(path)
    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.error(f"Failed to open PDF: {path}. Error: {e}")
        return chunks

    for i in range(len(doc)):
        page_no = i + 1
        text = (doc[i].get_text("text") or "").strip()
        if not text:
            continue

        words = text.split()
        MAX_WORDS = 600
        if len(words) <= MAX_WORDS:
            chunks.append(Chunk(text=text, source_file=base, doc_type=doc_type,
                          page=page_no, chunk_id=f"{base}:p{page_no}:c0"))
        else:
            part = 0
            for start in range(0, len(words), MAX_WORDS):
                seg = " ".join(words[start:start + MAX_WORDS]).strip()
                if seg:
                    chunks.append(Chunk(text=seg, source_file=base, doc_type=doc_type,
                                  page=page_no, chunk_id=f"{base}:p{page_no}:c{part}"))
                    part += 1

    logger.info(f"Parsed {len(chunks)} chunks from PDF {base} ({doc_type})")
    return chunks


def parse_html_to_chunks(path: str) -> List[Chunk]:
    """Lightweight table extraction first; else plain text; split every ~600 words."""
    chunks: List[Chunk] = []
    base = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
    except Exception as e:
        logger.error(f"Failed to open HTML: {path}. Error: {e}")
        return chunks

    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Try table rows as key:value lines for better signals
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            key = tds[0].get_text(strip=True)
            val = tds[1].get_text(" ", strip=True)
            if key and val:
                rows.append(f"{key}: {val}")

    if rows:
        text = "\n".join(rows)
    else:
        text = soup.get_text("\n", strip=True)

    words = text.split()
    MAX_WORDS = 600
    if len(words) <= MAX_WORDS:
        chunks.append(Chunk(text=text, source_file=base,
                      doc_type=DOC_HTML, page=None, chunk_id=f"{base}:c0"))
    else:
        part = 0
        for start in range(0, len(words), MAX_WORDS):
            seg = " ".join(words[start:start + MAX_WORDS]).strip()
            if seg:
                chunks.append(Chunk(text=seg, source_file=base,
                              doc_type=DOC_HTML, page=None, chunk_id=f"{base}:c{part}"))
                part += 1

    logger.info(f"Parsed {len(chunks)} chunks from HTML {base}")
    return chunks


def detect_doc_type_for_pdf(path: str) -> str:
    base = os.path.basename(path)
    try:
        doc = fitz.open(path)
        first_text = (doc[0].get_text("text")
                      or "").strip() if len(doc) > 0 else ""
    except Exception:
        first_text = ""
    return DOC_ADDENDUM if is_addendum_by_heuristics(base, first_text) else DOC_BASE


# =========================
# Embeddings & In-memory Index
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    """Call OpenAI embeddings; return NxD float32."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    vecs: List[List[float]] = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.asarray(vecs, dtype=np.float32)


class VectorIndex:
    def __init__(self) -> None:
        self.meta: List[Chunk] = []
        self.vecs: Optional[np.ndarray] = None

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        embs = embed_texts([c.text for c in chunks])
        if self.vecs is None:
            self.vecs = embs
        else:
            self.vecs = np.vstack([self.vecs, embs])
        self.meta.extend(chunks)
        logger.info(f"Indexed {len(chunks)} chunks (total {len(self.meta)}).")

    def search(self, query: str, top_k: int) -> List[Tuple[float, Chunk]]:
        if self.vecs is None or not len(self.meta):
            return []
        q = embed_texts([query])[0]
        sims = (self.vecs @ q) / (np.linalg.norm(self.vecs, axis=1)
                                  * (np.linalg.norm(q) + 1e-9))
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.meta[i]) for i in idx]


# =========================
# Retrieval
# =========================
def k_for_field(field: str) -> int:
    if field in {"Bid Number", "Title", "Due Date"}:
        return K_MAP["simple"]
    elif field in {"Bid Submission Type", "Term of Bid", "Delivery Date", "Payment Terms", "contact_info", "company_name"}:
        return K_MAP["medium"]
    else:
        # docs required, cooperatives, products/specs, summary, etc.
        return K_MAP["complex"]


def retrieve_snippets(index: VectorIndex) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: { field: [ {text_with_label, source_file, doc_type, page}, ... ] }
    text_with_label is prefixed with [SOURCE: <DOC_TYPE> | FILE: <name> | PAGE: n]
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for field, cues in FIELD_QUERIES.items():
        k = k_for_field(field)
        bag: List[Tuple[float, Chunk]] = []
        for cue in cues:
            bag.extend(index.search(cue, top_k=max(
                1, k // max(1, len(cues))) + 1))

        seen = set()
        ranked: List[Dict[str, Any]] = []
        for score, ch in sorted(bag, key=lambda t: -t[0]):
            key = (ch.source_file, ch.page, ch.doc_type, ch.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            label = f"[SOURCE: {ch.doc_type} | FILE: {ch.source_file} | PAGE: {ch.page if ch.page is not None else '-'}]"
            ranked.append({
                "score": score,
                "source_file": ch.source_file,
                "doc_type": ch.doc_type,
                "page": ch.page,
                "chunk_id": ch.chunk_id,  # internal/debug only
                "text_with_label": f"{label}\n{ch.text}"
            })
            if len(ranked) >= k:
                break

        out[field] = ranked

        # Keep logs lean: don’t print long content
        if ranked and logger.isEnabledFor(logging.DEBUG):
            tops = [r["text_with_label"].split("\n")[0] for r in ranked[:2]]
            logger.debug(f"Retrieved top-{len(ranked)} for '{field}': {tops}")
    return out


# =========================
# LLM call (single-pass) – SDK 2.6.0 compatible
# =========================
def call_llm_single_pass(prompt: str) -> Dict[str, Any]:
    """
    Compatible with openai==2.6.0:
    - Uses client.responses.create without response_format
    - Extracts text via response.output_text or fallbacks
    - Parses JSON, including fenced code blocks
    """
    def _extract_text(resp) -> str:
        txt = getattr(resp, "output_text", None)
        if txt:
            return txt
        try:
            # Sometimes available in nested content
            return resp.output[0].content[0].text
        except Exception:
            return str(resp)

    def _extract_json(text: str) -> Dict[str, Any]:
        # 1) direct JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2) fenced ```json ... ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            return json.loads(cand)
        # 3) first {...} blob
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("Could not parse JSON from model output")

    # Attempt 1
    try:
        r = client.responses.create(model=LLM_MODEL, input=prompt)
        txt = _extract_text(r)
        return _extract_json(txt)
    except Exception as e:
        logger.warning(f"LLM JSON parse failed (attempt 1): {e}")

    # Attempt 2 (nudge)
    try:
        r = client.responses.create(
            model=LLM_MODEL,
            input=prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No commentary or code fences."
        )
        txt = _extract_text(r)
        return _extract_json(txt)
    except Exception as e:
        logger.error(f"LLM JSON parse failed (attempt 2): {e}")

    raise RuntimeError(
        "Failed to obtain valid JSON from LLM after 2 attempts.")


# =========================
# Provenance (simple)
# =========================
def build_simple_provenance(
    final_data: Dict[str, Any],
    snippets: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each populated field, record one 'final' entry using the highest-precedence source
    seen in retrieved snippets. Arrays get a single final entry (no per-item tracking).
    """
    prov: Dict[str, List[Dict[str, Any]]] = {}
    precedence = [DOC_ADDENDUM, DOC_BASE, DOC_HTML]
    for field, value in final_data.items():
        if value in (None, [], ""):
            continue
        snips = snippets.get(field, [])
        if not snips:
            continue
        chosen = None
        for dt in precedence:
            chosen = next((s for s in snips if s["doc_type"] == dt), None)
            if chosen:
                break
        if not chosen:
            chosen = snips[0]
        prov[field] = [{
            "value": value,
            "source": chosen["source_file"],
            "page": chosen["page"],
            "final": True
        }]
    return prov


# =========================
# Normalize & validate
# =========================
def normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize LLM output before schema validation:
    - Ensure array fields are true List[str] (drop nulls/empties, dedupe)
    - List fields also sanitized
    - Product Specification coerced to list[dict] and pruned of nulls
    Then validate with Pydantic and return keys in assignment order.
    """
    raw = dict(raw)  # shallow copy

    # Clean array-of-strings product fields
    for fld in ("Product", "Model_no", "Part_no"):
        raw[fld] = _coerce_string_list(raw.get(fld))

    # Clean list fields (strings)
    for fld in ("Any Additional Documentation Required", "Contract or Cooperative to use"):
        raw[fld] = _coerce_string_list(raw.get(fld))

    # contact_info: force to a simple string if model returned a list/object
    ci = raw.get("contact_info")
    if isinstance(ci, list):
        # join non-empty stringy parts
        ci_list = _coerce_string_list(ci) or []
        raw["contact_info"] = ", ".join(ci_list) if ci_list else None
    elif isinstance(ci, (int, float)):
        raw["contact_info"] = str(ci)
    elif isinstance(ci, dict):
        # best-effort flatten
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

    # Product Specification: keep only list of objects (dicts), drop nulls/empties
    ps = raw.get("Product Specification")
    if ps is None:
        pass
    elif isinstance(ps, list):
        cleaned = []
        for item in ps:
            if isinstance(item, dict) and item:
                cleaned.append(item)
        raw["Product Specification"] = cleaned or None
    else:
        # If it's a single dict, wrap; if string, ignore (schema expects list[dict] or null)
        if isinstance(ps, dict) and ps:
            raw["Product Specification"] = [ps]
        else:
            raw["Product Specification"] = None

    # Now validate against schema
    try:
        model = AssignmentSchema(**raw)
    except ValidationError as ve:
        logger.error("Schema validation error after sanitation. "
                     "Most likely due to unexpected shapes in model output.")
        raise

    data = json.loads(model.model_dump_json(by_alias=True))
    return {k: data.get(k, None) for k in ASSIGNMENT_KEYS}


# =========================
# Main / CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="RFP Extractor – PDF+HTML → JSON via embeddings + LLM.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="List of input files for ONE bid (PDF(s) + HTML).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--bid_id", required=True,
                    help="Bid identifier for output filenames.")
    ap.add_argument("--provenance", choices=["on", "off"],
                    default="on", help="Write provenance sidecar JSON.")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # Validate inputs
    pdf_paths = [p for p in args.inputs if p.lower().endswith(".pdf")]
    html_paths = [p for p in args.inputs if p.lower().endswith(".html")]

    if not pdf_paths:
        logger.error("No PDF files supplied. Provide at least a base RFP PDF.")
        sys.exit(2)

    exit_code = 0
    if not html_paths:
        logger.error(
            f"HTML file missing for bid {args.bid_id}. Assignment requires HTML + PDF. "
            f"Attempting extraction from PDF only."
        )
        exit_code = 1

    logger.info(f"Starting extraction for bid {args.bid_id}")
    logger.info(f"PDFs: {len(pdf_paths)} | HTMLs: {len(html_paths)}")

    # Parse PDFs with doc_type detection
    all_chunks: List[Chunk] = []
    for pdf in pdf_paths:
        dtype = detect_doc_type_for_pdf(pdf)
        chs = parse_pdf_to_chunks(pdf, dtype)
        all_chunks.extend(chs)
        logger.info(f"{os.path.basename(pdf)} classified as {dtype}")

    # Parse HTMLs
    for html in html_paths:
        all_chunks.extend(parse_html_to_chunks(html))

    if not all_chunks:
        logger.error("No text could be extracted from the provided inputs.")
        sys.exit(2)

    # Index
    t_embed = time.time()
    index = VectorIndex()
    index.add(all_chunks)
    logger.info(f"Embeddings + index built in {time.time() - t_embed:.2f}s")

    # Retrieve
    t_ret = time.time()
    snippets = retrieve_snippets(index)
    logger.info(f"Retrieval completed in {time.time() - t_ret:.2f}s")

    # Build prompt (annotated snippets). No prompt preview to keep logs small.
    prompt = build_single_pass_prompt(snippets)

    # LLM
    t_llm = time.time()
    try:
        llm_json = call_llm_single_pass(prompt)
    except RuntimeError as e:
        logger.error(f"LLM failed to return valid JSON: {e}")
        sys.exit(2)
    logger.info(f"LLM extraction completed in {time.time() - t_llm:.2f}s")

    # Normalize & validate
    try:
        final_json = normalize_output(llm_json)
    except ValidationError:
        sys.exit(2)

    # Provenance
    provenance_json: Dict[str, Any] = {}
    if args.provenance == "on":
        provenance_json = build_simple_provenance(final_json, snippets)

    # Write outputs
    out_main = os.path.join(args.out, f"{args.bid_id}.json")
    with open(out_main, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote: {out_main}")

    if args.provenance == "on":
        out_prov = os.path.join(args.out, f"{args.bid_id}.provenance.json")
        with open(out_prov, "w", encoding="utf-8") as f:
            json.dump(provenance_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote: {out_prov}")

    # Summary
    try:
        populated = sum(1 for v in final_json.values()
                        if v not in (None, [], ""))
        logger.info("=" * 60)
        logger.info(
            f"Extraction Summary for Bid {final_json.get('Bid Number') or args.bid_id}")
        logger.info("=" * 60)
        logger.info(f"Fields populated: {populated}/{len(ASSIGNMENT_KEYS)}")
        if isinstance(final_json.get("Product"), list):
            logger.info(f"Products found: {len(final_json['Product'])}")
        logger.info("=" * 60)
    except Exception:
        pass

    logger.info(f"Total runtime: {time.time() - t0:.2f}s")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
