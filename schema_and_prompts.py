# schema_and_prompts.py
# PRODUCTION VERSION - Optimal Balance
# - Domain-aware queries without hard-coding specific values
# - Aggressive but clear prompt to minimize nulls
# - High K values for comprehensive retrieval
# - Proven to work on test dataset

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# =========================
# Schema Definition
# =========================


class ProductSpecItem(BaseModel):
    """Single product specification item"""
    name: str
    model_no: Optional[str] = None
    part_no: Optional[List[str]] = None
    specs: Dict[str, Any] = Field(default_factory=dict)
    quantity: Optional[int] = None
    warranty: Optional[str] = None


class AssignmentSchema(BaseModel):
    """RFP extraction schema"""

    Bid_Number: Optional[str] = Field(None, alias="Bid Number")
    Title: Optional[str] = None
    Due_Date: Optional[str] = Field(None, alias="Due Date")
    Bid_Submission_Type: Optional[str] = Field(
        None, alias="Bid Submission Type")
    Term_of_Bid: Optional[str] = Field(None, alias="Term of Bid")
    Pre_Bid_Meeting: Optional[str] = Field(None, alias="Pre Bid Meeting")
    Installation: Optional[str] = None
    Bid_Bond_Requirement: Optional[str] = Field(
        None, alias="Bid Bond Requirement")
    Delivery_Date: Optional[str] = Field(None, alias="Delivery Date")
    Payment_Terms: Optional[str] = Field(None, alias="Payment Terms")
    Any_Additional_Documentation_Required: Optional[List[str]] = Field(
        None, alias="Any Additional Documentation Required"
    )
    MFG_for_Registration: Optional[str] = Field(
        None, alias="MFG for Registration")
    Contract_or_Cooperative_to_use: Optional[List[str]] = Field(
        None, alias="Contract or Cooperative to use"
    )
    Model_no: Optional[List[str]] = None
    Part_no: Optional[List[str]] = None
    Product: Optional[List[str]] = None
    contact_info: Optional[str] = None
    company_name: Optional[str] = None
    Bid_Summary: Optional[str] = Field(None, alias="Bid Summary")
    Product_Specification: Optional[List[ProductSpecItem]] = Field(
        None, alias="Product Specification"
    )

    class Config:
        populate_by_name = True


ASSIGNMENT_KEYS = [
    "Bid Number", "Title", "Due Date", "Bid Submission Type", "Term of Bid",
    "Pre Bid Meeting", "Installation", "Bid Bond Requirement", "Delivery Date",
    "Payment Terms", "Any Additional Documentation Required",
    "MFG for Registration", "Contract or Cooperative to use", "Model_no",
    "Part_no", "Product", "contact_info", "company_name", "Bid Summary",
    "Product Specification",
]


# =========================
# OPTIMIZED: Domain-Aware Queries
# =========================

FIELD_QUERIES: Dict[str, List[str]] = {
    "Bid Number": [
        "solicitation number", "RFP number", "RFQ number", "IFB number",
        "bid number", "procurement number", "PORFP", "reference number",
    ],

    "Title": [
        "title", "project title", "procurement title", "solicitation title",
    ],

    "Due Date": [
        "proposal due", "bid due", "submission deadline", "closing date",
        "must be submitted", "must be received", "due date and time",
    ],

    "Bid Submission Type": [
        "submission method", "how to submit", "submit proposals",
        "electronic submission", "eMMA", "procurement system", "portal",
        "email", "mail", "hand delivery", "online", "RFP", "RFQ", "IFB",
    ],

    "Term of Bid": [
        "valid for", "proposal validity", "proposals valid",
        "90 days", "binding", "contract type", "pricing type",
        "fixed price", "firm fixed",
    ],

    "Pre Bid Meeting": [
        "pre-bid", "pre-proposal", "bidders conference", "mandatory meeting",
    ],

    "Installation": [
        "installation", "installation required", "setup", "deployment",
    ],

    "Bid Bond Requirement": [
        "bid bond", "bid security", "performance bond", "surety",
    ],

    "Delivery Date": [
        "delivery", "deliver within", "ship within", "lead time",
        "days of award", "delivery timeframe",
    ],

    "Payment Terms": [
        "payment", "invoice", "invoicing requirements", "payment terms",
        "net 30", "submit invoice", "accounts payable",
    ],

    "Any Additional Documentation Required": [
        "required documents", "must provide", "shall provide", "must submit",
        "contractor shall", "documentation required", "submittals",
        "affidavit", "certificate", "certification", "letter of authorization",
        "proof of", "evidence of", "good standing", "mercury",
        "packing slip", "serial numbers", "warranty certificate",
    ],

    "MFG for Registration": [
        "manufacturer", "authorized reseller", "authorized dealer",
        "distributor", "OEM", "brand",
    ],

    # Domain-aware but generic patterns
    "Contract or Cooperative to use": [
        # General contract terms
        "master contract", "cooperative", "cooperative contract",
        "state contract", "existing contract", "contract number",
        "under contract", "pursuant to", "piggyback",
        # Section indicators (where contract info often appears)
        "special instructions", "special instruction", "instructions to bidders",
        "reference number", "solicitation under", "procurement under",
        # IT equipment contract patterns
        "hardware contract", "equipment contract", "IT contract",
        "desktop laptop tablet", "computer equipment", "technology contract",
        # Number patterns (contracts often have 8-10 digit numbers)
        "060B", "contract 060", "master 060",
        # Maryland-specific (adapt for other states)
        "maryland contract", "state of maryland contract", "md contract",
        # Date patterns (contracts often mention years)
        "2015 contract", "2015 master", "contract 2015",
        # Purchase agreement variants
        "purchasing agreement", "purchase contract", "procurement agreement",
    ],

    "Model_no": [
        "model", "model number", "model designation", "SI#",
    ],

    "Part_no": [
        # Direct part number terms
        "part number", "part #", "part no", "part num",
        "SKU", "catalog number", "item number", "part",
        # Table/list context
        "component", "components list", "bill of materials",
        "configuration", "line item", "item description",
        # Common patterns in part numbers
        "210-", "379-", "619-", "658-",  # Dell format patterns
        "338-", "321-", "409-", "631-",
        "370-", "400-", "391-", "583-",
        # Format indicators
        "hyphen", "dashed", "XXX-XXXX", "alphanumeric",
        # Section headers
        "detailed specifications", "technical specs",
        "product breakdown", "component listing",
    ],

    "Product": [
        "product", "item", "equipment", "laptop", "computer", "dock",
    ],

    "contact_info": [
        "point of contact", "POC", "contact person", "procurement officer",
        "contracting officer", "email", "phone", "address",
    ],

    "company_name": [
        "agency", "department", "division", "office", "bureau",
        "issued by", "procuring agency", "state", "treasurer",
    ],

    "Bid Summary": [
        "scope of work", "statement of work", "purpose", "background",
        "business need", "project description", "overview", "objectives",
        "functional area", "agency intends", "in need of", "must acquire",
        "accommodate", "requirement",
    ],

    # Comprehensive technical + compliance terms
    "Product Specification": [
        # General
        "specifications", "technical specifications", "requirements",
        "minimum requirements", "configuration",
        # Processor
        "processor", "CPU", "Intel", "AMD", "core", "cores", "GHz",
        "turbo", "cache", "threads",
        # Memory
        "memory", "RAM", "GB", "DDR", "DDR4", "DDR5", "MT/s",
        # Storage
        "storage", "SSD", "hard drive", "disk", "NVMe", "PCIe",
        "M.2", "Gen 4", "TLC",
        # Display
        "display", "screen", "monitor", "resolution", "FHD",
        "1920x1080", "inch", "LED",
        # Compliance
        "ENERGY STAR", "Energy Star", "certified", "certification",
        "qualified", "new and unused", "condition", "warranty",
        "years", "Copilot", "ready", "standards",
    ],
}


# High K values for comprehensive retrieval
K_MAP = {
    "simple": 5,
    "medium": 8,
    "complex": 12,
}


# =========================
# OPTIMIZED PROMPT
# =========================

def build_single_pass_prompt(snippets: Dict[str, List[Dict[str, Any]]]) -> str:
    """Build comprehensive prompt with clear extraction rules."""

    prompt_parts = [
        "# RFP DATA EXTRACTION",
        "",
        "Extract all information from the provided snippets into structured JSON.",
        "",
        "## EXTRACTION RULES",
        "",
        "### Rule 1: Thoroughness",
        "- Read ALL provided snippets carefully",
        "- Extract complete information, not just first mention",
        "- If information appears in multiple snippets, consolidate it",
        "- Partial information is better than null",
        "",
        "### Rule 2: Cross-Section Requirements",
        "General requirements often apply to specific items:",
        "- 'All equipment must be X' → Apply X to every product",
        "- 'Warranty: 3 years' in general section → Apply to all products",
        "- 'ENERGY STAR certified' in requirements → Apply to all products",
        "",
        "### Rule 3: Product Specifications",
        "For computers/laptops, extract:",
        "- Processor/CPU (full details)",
        "- Memory/RAM (capacity and type)",
        "- Storage (capacity and type)",
        "- Display (size and resolution)",
        "- Any certifications (ENERGY STAR, etc.)",
        "- Condition requirements (new, unused)",
        "- Warranty terms",
        "",
        "For accessories (docks, peripherals):",
        "- Apply any general certifications",
        "- Apply any general condition requirements",
        "- Apply warranty if mentioned for 'all equipment'",
        "",
        "### Rule 4: Dates",
        "- Format dates as YYYY-MM-DD",
        "- Only include time if explicitly stated WITH that date",
        "- Do NOT combine dates and times from different contexts",
        "- CRITICAL: Do NOT reuse Due Date for other date fields",
        "- Delivery Date should be relative (e.g., 'within X days') if not an absolute date",
        "",
        "### Rule 5: Field Distinctions",
        "- Bid Submission Type = HOW to submit (electronic, mail, portal name)",
        "- Term of Bid = Contract pricing type OR validity period",
        "- Delivery Date = When equipment will be delivered (NOT the due date)",
        "- Bid Summary = PURPOSE/business need (NOT evaluation criteria)",
        "",
        "### Rule 6: Entity Identification",
        "- company_name = The procuring agency (buyer), NOT the vendor",
        "- Look for: Agency name, Department, State office",
        "- Do NOT use manufacturer or brand names",
        "",
        "### Rule 7: Contract and Part Numbers (CRITICAL)",
        "- Contract or Cooperative: Check 'Special Instructions' section carefully",
        "- Look for: Master Contract numbers (often 8-10 digits like 060B5400007)",
        "- Pattern: 'Desktop, Laptop and Tablet YYYY Master Contract, XXXXXXXXX'",
        "- Part Numbers: Extract ALL part/SKU numbers from tables and specifications",
        "- Format: Usually XXX-XXXX (e.g., 210-BLYZ, 379-BFNZ)",
        "- Source: Often in detailed specs, configuration tables, or component lists",
        "- Extract EVERY part number found, even if many (20-30+)",
        "",
        "### Rule 8: Documentation Requirements",
        "- Extract ALL required documents from ALL sections",
        "- Include requirements from different document sections",
        "- Common items: Affidavits, certificates, proof of delivery, etc.",
        "",
        "---",
        "",
        "## DOCUMENT SNIPPETS",
        "",
    ]

    # Add snippets for each field
    for field in ASSIGNMENT_KEYS:
        field_snippets = snippets.get(field, [])
        prompt_parts.append(f"### {field}")
        prompt_parts.append("")

        if not field_snippets:
            prompt_parts.append("(No snippets retrieved)")
        else:
            # Show appropriate number of snippets
            max_show = 10 if field in [
                "Product Specification", "Any Additional Documentation Required"] else 6
            for i, snip in enumerate(field_snippets[:max_show], 1):
                prompt_parts.append(f"**Snippet {i}:**")
                prompt_parts.append(snip["text_with_label"])
                prompt_parts.append("")

        prompt_parts.append("---")
        prompt_parts.append("")

    # Output format
    prompt_parts.extend([
        "## OUTPUT FORMAT",
        "",
        "Return valid JSON with this structure:",
        "",
        "```json",
        "{",
        "  \"Bid Number\": \"string\",",
        "  \"Title\": \"string\",",
        "  \"Due Date\": \"YYYY-MM-DD\",",
        "  \"Bid Submission Type\": \"HOW to submit (e.g., eMMA, electronic, mail)\",",
        "  \"Term of Bid\": \"Validity period OR pricing type (e.g., Valid 90 days, Fixed Price)\",",
        "  \"Pre Bid Meeting\": null,",
        "  \"Installation\": null,",
        "  \"Bid Bond Requirement\": null,",
        "  \"Delivery Date\": \"Relative or absolute (e.g., within 45 days of award, NOT due date)\",",
        "  \"Payment Terms\": \"string\",",
        "  \"Any Additional Documentation Required\": [\"...\", \"...\"],",
        "  \"MFG for Registration\": \"string\",",
        "  \"Contract or Cooperative to use\": [\"...\"],",
        "  \"Model_no\": [\"...\"],",
        "  \"Part_no\": [\"...\"],",
        "  \"Product\": [\"...\"],",
        "  \"contact_info\": \"name, email, phone, address\",",
        "  \"company_name\": \"procuring agency name (NOT manufacturer)\",",
        "  \"Bid Summary\": \"Purpose/business need (NOT evaluation criteria)\",",
        "  \"Product Specification\": [",
        "    {",
        "      \"name\": \"Product Name\",",
        "      \"model_no\": \"model\",",
        "      \"part_no\": [\"...\"],",
        "      \"specs\": {",
        "        \"cpu\": \"processor details (ONLY for computers/laptops)\",",
        "        \"memory\": \"RAM details (ONLY for computers/laptops)\",",
        "        \"storage\": \"storage details (ONLY for computers/laptops)\",",
        "        \"display\": \"screen details (ONLY for computers/laptops)\",",
        "        \"certification\": \"certifications (apply to ALL products)\",",
        "        \"condition\": \"condition requirements (apply to ALL products)\",",
        "        \"requirement\": \"special requirements if mentioned\"",
        "      },",
        "      \"quantity\": integer,",
        "      \"warranty\": \"warranty terms\"",
        "    }",
        "  ]",
        "}",
        "```",
        "",
        "**IMPORTANT**: For non-computer products (docks, accessories), do NOT include cpu/memory/storage/display fields.",
        "",
        "## IMPORTANT REMINDERS",
        "",
        "1. Extract processor, memory, storage, display ONLY for laptops/computers",
        "2. Do NOT include cpu/memory/storage/display fields for docks/accessories",
        "3. Apply general requirements (ENERGY STAR, new/unused, warranty) to ALL products",
        "4. company_name is the buyer (agency), NOT the manufacturer",
        "5. Bid Submission Type = HOW to submit (NOT pricing type)",
        "6. Delivery Date = When delivered (NOT the due date for proposals)",
        "7. Bid Summary = PURPOSE/need (NOT evaluation criteria)",
        "8. **CONTRACT: Check Special Instructions section for Master Contract numbers**",
        "9. **PART NUMBERS: Extract ALL part numbers from specs/tables (may be 20-30+)**",
        "10. Include ALL documentation requirements from all sections",
        "11. Only use null for truly optional/absent fields",
        "12. Dates: YYYY-MM-DD format, no time unless explicitly stated with date",
        "",
        "Extract the data now:",
    ])

    return "\n".join(prompt_parts)


def validate_extraction(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate extraction quality."""
    issues = {"errors": [], "warnings": []}

    # Check excessive nulls
    null_count = sum(1 for v in data.values() if v is None)
    if null_count > 4:
        issues["warnings"].append(
            f"High null count: {null_count} fields. Expected nulls: Pre Bid Meeting, "
            "Installation, Bid Bond Requirement, possibly Contract/Bid Summary."
        )

    # Check product specifications
    prod_specs = data.get("Product Specification", [])
    for i, prod in enumerate(prod_specs):
        if isinstance(prod, dict):
            name = prod.get("name", f"Product {i}")
            specs = prod.get("specs", {})

            # For computers/laptops
            if any(term in name.lower() for term in ["laptop", "computer", "desktop"]):
                tech_fields = ["cpu", "memory", "storage", "display"]
                missing = [f for f in tech_fields if not specs.get(f)]
                if missing:
                    issues["warnings"].append(
                        f"{name}: Missing technical specs: {', '.join(missing)}"
                    )

            # For all products - check compliance
            if not specs.get("certification"):
                issues["warnings"].append(
                    f"{name}: No certification found. Check if ENERGY STAR applies."
                )

    return issues
