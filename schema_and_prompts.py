# schema_and_prompts.py
# ---------------------
# Schema + queries + prompt builder for the RFP extractor.
# Matches the design we finalized:
# - Top-level keys exactly as the assignment expects
# - Arrays for Product / Model_no / Part_no
# - "Product Specification" as an array of minimal objects
# - contact_info is a flat string
# - Single-pass prompt with doc-type precedence instructions
# - Curated FIELD_QUERIES and fixed K tiers (3/6/10)

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =========================
# Assignment JSON Schema
# =========================

class AssignmentSchema(BaseModel):
    # NOTE: We keep aliases exactly matching the assignment’s fields.
    Bid_Number: Optional[str] = Field(default=None, alias="Bid Number")
    Title: Optional[str] = None
    # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
    Due_Date: Optional[str] = Field(default=None, alias="Due Date")
    Bid_Submission_Type: Optional[str] = Field(
        default=None, alias="Bid Submission Type")
    Term_of_Bid: Optional[str] = Field(default=None, alias="Term of Bid")
    Pre_Bid_Meeting: Optional[str] = Field(
        default=None, alias="Pre Bid Meeting")
    Installation: Optional[str] = None
    Bid_Bond_Requirement: Optional[str] = Field(
        default=None, alias="Bid Bond Requirement")
    Delivery_Date: Optional[str] = Field(default=None, alias="Delivery Date")
    Payment_Terms: Optional[str] = Field(default=None, alias="Payment Terms")

    Any_Additional_Documentation_Required: Optional[List[str]] = Field(
        default=None, alias="Any Additional Documentation Required"
    )
    MFG_for_Registration: Optional[str] = Field(
        default=None, alias="MFG for Registration")
    Contract_or_Cooperative_to_use: Optional[List[str]] = Field(
        default=None, alias="Contract or Cooperative to use"
    )

    # Product-related (arrays, per our decision)
    Model_no: Optional[List[str]] = None
    Part_no: Optional[List[str]] = None
    Product: Optional[List[str]] = None

    # contact_info is a flat string ("Name, email@x, 123-456-7890")
    contact_info: Optional[str] = None

    company_name: Optional[str] = None
    Bid_Summary: Optional[str] = Field(default=None, alias="Bid Summary")

    # Minimal item structure for specs (not strictly enforced; LLM will populate)
    Product_Specification: Optional[List[Dict[str, Any]]] = Field(
        default=None, alias="Product Specification")

    class Config:
        populate_by_name = True
        extra = "ignore"


# Output order (exact keys)
ASSIGNMENT_KEYS: List[str] = [
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
]


# =========================
# Retrieval config
# =========================

# Curated cue phrases per field (short and effective).
FIELD_QUERIES: Dict[str, List[str]] = {
    "Bid Number": [
        "bid number", "solicitation number", "RFP number", "project number", "PORFP", "SOURCING", "BPM"
    ],
    "Title": [
        "title", "solicitation title", "project title", "name of solicitation"
    ],
    "Due Date": [
        "due date", "closing date", "submission deadline", "proposals due", "closing time"
    ],
    "Bid Submission Type": [
        "submission method", "submit via", "portal", "eMMA", "iSupplier", "electronic only", "no email", "no fax"
    ],
    "Term of Bid": [
        "term", "contract term", "duration", "renewal", "initial term"
    ],
    "Pre Bid Meeting": [
        "pre-bid", "pre proposal", "pre-proposal", "site visit", "Teams meeting", "prebid conference"
    ],
    "Installation": [
        "installation", "deployment", "white glove", "asset tagging", "etching", "staging"
    ],
    "Bid Bond Requirement": [
        "bid bond", "bond", "bonding", "security deposit", "bid security"
    ],
    "Delivery Date": [
        "delivery within", "delivery schedule", "lead time", "ship within", "delivery date"
    ],
    "Payment Terms": [
        "payment terms", "invoice", "net", "accounts payable", "payment schedule"
    ],
    "Any Additional Documentation Required": [
        "affidavit", "certificate", "form", "LOA", "insurance", "mercury", "EDGAR", "1295", "interested parties"
    ],
    "MFG for Registration": [
        "manufacturer registration", "MFG registration", "vendor registration with manufacturer"
    ],
    "Contract or Cooperative to use": [
        "cooperative", "EPCNT", "CTPA", "master contract", "state contract", "piggyback"
    ],
    "Model_no": [
        "model", "model number", "model #", "Latitude", "Chromebook", "ThinkPad"
    ],
    "Part_no": [
        "SKU", "part number", "PN", "Dell part", "spare", "catalog number"
    ],
    "Product": [
        "devices", "items", "laptops", "desktops", "tablets", "monitors", "docks"
    ],
    "contact_info": [
        "contact", "buyer", "procurement officer", "email", "phone", "address"
    ],
    "company_name": [
        "issuing agency", "organization", "district", "department", "office"
    ],
    "Bid Summary": [
        "purpose", "intent", "scope", "overview", "objective"
    ],
    "Product Specification": [
        "specifications", "technical requirements", "minimum specifications", "line items", "configuration"
    ],
}

# K tiers (fixed)
K_MAP: Dict[str, int] = {
    "simple": 3,    # IDs, title, due date
    "medium": 6,    # submission type, term, delivery, payment, contacts, company
    "complex": 10,  # docs required, cooperatives, products/specs, bid summary
}


# =========================
# Prompt builder (single-pass)
# =========================

_FEW_SHOT_EXAMPLE = r'''
# Example output format for product-related keys (illustrative)
"Product": ["Dell Latitude 5550", "Dell WD22TB4 Dock"],
"Model_no": ["Latitude 5550"],
"Part_no": ["210-BLYZ", "379-BFNZ", "WD22TB4"],
"Product Specification": [
  {
    "name": "Dell Latitude 5550",
    "model_no": "Latitude 5550",
    "part_no": ["210-BLYZ"],
    "specs": { "cpu": "Intel Core Ultra 5", "memory": "16GB", "storage": "256GB NVMe" },
    "quantity": 30,
    "warranty": "3 years"
  }
]
'''.strip()

_FIELD_GUIDE = {
    "Bid Number": "Primary solicitation identifier (e.g., PORFP #, Sourcing #, BPM #). String.",
    "Title": "Official solicitation title. String.",
    "Due Date": "Submission deadline. Use YYYY-MM-DD or YYYY-MM-DD HH:MM.",
    "Bid Submission Type": "How responses must be submitted (e.g., eMMA only, iSupplier, no email/fax).",
    "Term of Bid": "Contract term/duration, including renewals.",
    "Pre Bid Meeting": "Date/time/location or virtual link of pre-bid meeting if present; else null.",
    "Installation": "Deployment/installation requirements if any; else null.",
    "Bid Bond Requirement": "Bid bond requirement text; else null.",
    "Delivery Date": "Delivery timeline (e.g., 'within 45 days of award'); else null.",
    "Payment Terms": "Payment terms/invoicing (e.g., NET30); else null.",
    "Any Additional Documentation Required": "List of required forms/affidavits/certificates.",
    "MFG for Registration": "Manufacturer registration requirement; else null.",
    "Contract or Cooperative to use": "Cooperatives or master contracts permitted. List.",
    "Model_no": "Array of model names/numbers across items.",
    "Part_no": "Array of part numbers/SKUs across items.",
    "Product": "Array of product names/items.",
    "contact_info": "Single string: 'Name, email, phone, address' (as available).",
    "company_name": "Issuing agency/organization name.",
    "Bid Summary": "2–3 sentence plain-English summary of purpose/scope.",
    "Product Specification": "Array of objects with {name, model_no, part_no[], specs{...}, quantity, warranty} where available.",
}


def _field_guide_text() -> str:
    lines = []
    for k in ASSIGNMENT_KEYS:
        g = _FIELD_GUIDE.get(k, "")
        lines.append(f'- "{k}": {g}')
    return "\n".join(lines)


def build_single_pass_prompt(snippets_by_field: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Build a single-pass prompt that:
      - Lists the exact keys to output
      - Provides a compact field guide
      - Provides a small few-shot example for Product fields
      - Injects annotated snippets with [SOURCE: ADDENDUM|BASE PDF|HTML | FILE: ... | PAGE: ...]
      - Instructs precedence: ADDENDUM > BASE PDF > HTML for hard fields
      - Requires JSON-only output; unknowns as null; arrays for product fields
    """
    header = f"""
You are an expert RFP information extractor. Read the document excerpts and produce ONE JSON object
with EXACTLY these top-level keys in English (unknown values must be null):

{ASSIGNMENT_KEYS}

Field guide:
{_field_guide_text()}

Formatting rules:
- Output JSON only (no markdown, no comments).
- Dates must be in YYYY-MM-DD, or YYYY-MM-DD HH:MM if time is included.
- "Product", "Model_no", "Part_no" must be arrays of strings (or null if unknown).
- "Product Specification" must be an array of objects (or null), each like:
  {{"name": "...", "model_no": "...", "part_no": ["..."], "specs": {{"key": "value"}}, "quantity": 10, "warranty": "..."}}.
- contact_info must be a single string ("Name, email, phone, address") using whatever parts are available.

Precedence for conflicts (for these hard fields: "Due Date", "Delivery Date", "Bid Submission Type", "Term of Bid"):
- Prioritize ADDENDUM over BASE PDF over HTML.
  (You will see every snippet prefixed with its source label.)

Few-shot example for product fields (illustrative only, not a gold label):
{_FEW_SHOT_EXAMPLE}

=== BEGIN CONTEXT SNIPPETS ===
""".strip()

    # Build labeled snippets, grouped by field
    body_parts: List[str] = []
    for field, items in snippets_by_field.items():
        if not items:
            continue
        body_parts.append(f"\n<FIELD name='{field}'>")
        for it in items:
            # it requires: text_with_label (already prefixed), source_file, doc_type, page
            body_parts.append(it["text_with_label"])
        body_parts.append(f"</FIELD>")

    footer = """
=== END CONTEXT SNIPPETS ===

Now produce the single JSON object with exactly the keys listed above.
If a field cannot be determined from the context, set it to null.
Remember precedence rules for the hard fields and keep product-related fields as arrays.
""".strip()

    return header + "\n" + "\n".join(body_parts) + "\n" + footer
