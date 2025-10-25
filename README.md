# 📚 **RFP Data Extraction System**

An advanced **Retrieval-Augmented Generation (RAG)** system for automated extraction of structured data from Request for Proposal (RFP) documents. The system processes multi-format procurement documents (PDFs, HTML) and extracts critical bidding information with high accuracy, comprehensive provenance tracking, and full transparency.

---

## 📋 **Table of Contents**

- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Provenance & Transparency](#provenance--transparency)
- [Technical Implementation](#technical-implementation)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)

---

## 📁 **Project Structure**

```
rfp-extractor/
├── rfp_extractor.py              # Main extraction pipeline
├── schema_and_prompts.py         # Schema definitions & prompt templates
├── .env                          # Environment configuration (not in repo)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── logs/
│   └── rfp_extractor.log         # Execution logs
└── outputs/
    ├── {bid_id}.json             # Structured extraction
    ├── {bid_id}_provenance.json  # Complete source tracking
    └── {bid_id}_debug_prompt.txt # Debug output (optional)
```

---

## 🚀 **Installation & Usage**

### **Prerequisites**

- Python 3.9 or higher
- OpenAI API key

### **Installation**

1. **Clone the repository**
```bash
git clone <repository-url>
cd rfp-extractor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install openai>=1.0.0 PyMuPDF>=1.23.0 beautifulsoup4>=4.12.0 \
            numpy>=1.24.0 python-dotenv>=1.0.0 pydantic>=2.0.0
```

3. **Configure environment**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-api-key-here
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

### **Basic Usage**

Extract data from RFP documents:
```bash
python rfp_extractor.py \
  --inputs document1.pdf document2.pdf document3.html \
  --out outputs \
  --bid_id E20P4600040 \
  --provenance on
```

### **Command-Line Options**

| Option | Description | Required |
|--------|-------------|----------|
| `--inputs` | Input PDF/HTML files (space-separated) | Yes |
| `--out` | Output directory | Yes |
| `--bid_id` | Bid identifier (used in output filenames) | Yes |
| `--provenance` | Enable provenance tracking (`on` or `off`) | No (default: `off`) |
| `--debug` | Save debug prompt with retrieval context | No |

### **With Debug Mode** (Recommended for Verification)

```bash
python rfp_extractor.py \
  --inputs PORFP.pdf specs.pdf affidavit.pdf listing.html \
  --out outputs \
  --bid_id E20P4600040 \
  --provenance on \
  --debug
```

### **Output Files**

After execution, the following files are generated:

| File | Description |
|------|-------------|
| `{bid_id}.json` | Structured extraction with all 20 fields |
| `{bid_id}_provenance.json` | Complete source tracking (if enabled) |
| `{bid_id}_debug_prompt.txt` | Full retrieval context (if `--debug` used) |
| `logs/rfp_extractor.log` | Execution logs with warnings/errors |

### **Example Output Structure**

**Extraction JSON** (`E20P4600040.json`):
```json
{
  "Bid Number": "E20P4600040",
  "Title": "Dell Laptops w/Extended Warranty",
  "Due Date": "2024-06-10",
  "Product Specification": [
    {
      "name": "Dell Latitude 5550",
      "specs": {
        "cpu": "Intel Core Ultra 5 125U...",
        "memory": "16 GB DDR5",
        ...
      }
    }
  ]
}
```

**Provenance JSON** (`E20P4600040_provenance.json`):
```json
{
  "Bid Number": [
    {
      "value": "E20P4600040",
      "source": "PORFP_Dell_Laptop_Final.pdf",
      "page": 1,
      "confidence_score": 0.558
    }
  ]
}
```

---

## 🏗️ **System Architecture**

### **Multi-Stage Pipeline**

```
┌─────────────────┐
│  Input Files    │
│  (PDF, HTML)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Document       │
│  Parsing        │
│  - Chunking     │
│  - Cleaning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │
│  Generation     │
│  (OpenAI)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Multi-Query    │
│  Retrieval      │
│  - Field-specific│
│  - Adaptive K   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM            │
│  Extraction     │
│  (GPT-4o-mini)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation &   │
│  Normalization  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Provenance     │
│  Tracking       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Structured     │
│  JSON Output    │
└─────────────────┘
```

---

## ✨ **Core Features**

### **1. Multi-Document Processing**
- **PDF Parsing**: Robust extraction with PyMuPDF, handling complex layouts and tables
- **HTML Processing**: Clean extraction with BeautifulSoup, removing scripts and styles
- **Document Classification**: Automatic identification of base documents vs. addenda
- **Cross-Platform Compatibility**: Platform-specific optimizations for Windows and Linux

### **2. Advanced Retrieval Strategy**
- **Multi-Query Retrieval**: 4-23 semantic queries per field for comprehensive coverage
- **Adaptive K Values**: Dynamic chunk retrieval based on field complexity
- **Pattern-Based Matching**: Specialized queries for structured data (contract numbers, part numbers)
- **Section-Aware Retrieval**: Targeted queries for specific document sections

### **3. Intelligent Extraction**
- **Field-Specific Prompts**: Tailored instructions for each data type
- **Cross-Section Aggregation**: Applies general requirements to specific items
- **Structured Output**: Pydantic schemas ensure data consistency
- **Validation Pipeline**: Post-extraction validation with comprehensive error logging

### **4. Data Quality Assurance**
- **Normalization**: Phone numbers, emails, dates formatted consistently
- **Deduplication**: Automatic removal of duplicate entries
- **UTF-8 Handling**: Smart quote and special character normalization
- **Null Management**: Contextual handling of optional vs. missing fields

---

## 🔍 **Provenance & Transparency**

### **Why Provenance Matters**

In procurement contexts, **transparency and auditability are paramount**. Every extracted data point must be verifiable against source documents to ensure:
- **Accountability**: Stakeholders can verify extraction accuracy
- **Compliance**: Meets procurement transparency requirements
- **Debugging**: Enables systematic improvement of extraction quality
- **Trust**: Builds confidence in automated systems

### **Our Commitment to Transparency**

This system was built with an unwavering commitment to transparency at every stage of the data extraction pipeline. We believe that automated systems handling critical procurement data must be fully auditable and that every piece of extracted information should be traceable to its source.

### **Comprehensive Provenance System**

Our system implements **multi-level provenance tracking** that goes beyond basic source attribution:

#### **1. Primary Source Attribution**
Every field includes complete source information:
- **Source File**: Exact document containing the data
- **Page Number**: Precise location (1-indexed for PDFs, null for HTML)
- **Confidence Score**: Semantic similarity score from retrieval (0.0-1.0)
- **Final Flag**: Indicates this is the authoritative source

Example structure:
```json
{
  "value": "E20P4600040",
  "source": "PORFP_Dell_Laptop_Final.pdf",
  "page": 1,
  "final": true,
  "confidence_score": 0.558
}
```

#### **2. Contributing Sources**
For aggregated data (arrays, lists), we track **all contributing sources**, not just the primary one. This enables stakeholders to:
- **Cross-Document Validation**: Verify data consistency across multiple files
- **Information Synthesis**: Track how data was aggregated from different sources
- **Quality Assessment**: Identify when data comes from authoritative vs. secondary sources

Each aggregated field includes a `contributing_sources` array showing every document and page that contributed to the final value.

#### **3. Document Type Classification**
Documents are classified into three categories with clear precedence:
- **ADDENDUM**: Updates or corrections (highest priority)
- **BASE_PDF**: Primary procurement documents (medium priority)
- **HTML**: Web-based listings or summaries (lowest priority)

This classification ensures that when information conflicts across sources, the most authoritative source takes precedence.

#### **4. Confidence Scoring**
Every extraction includes a confidence score derived from the semantic similarity between queries and retrieved chunks. This allows reviewers to:
- **Prioritize Verification**: Focus on low-confidence extractions
- **Assess Reliability**: Understand extraction certainty
- **Track Quality**: Monitor system performance over time

#### **5. Debug Mode for Deep Transparency**

Beyond standard provenance, the system offers a **complete transparency mode** via the `--debug` flag. This generates a comprehensive debug file containing:

- **All Retrieved Snippets**: Every chunk retrieved for each field
- **Complete LLM Prompt**: The exact instructions and context sent to the model
- **Source Labels**: Every snippet labeled with `[filename, page X]`
- **Retrieval Context**: Why each snippet was selected (query matches)

This level of transparency means that **every extraction decision can be fully reconstructed and audited** by independent reviewers. If a stakeholder questions why a particular value was extracted, they can examine the exact context the system used to make that decision.

### **Transparency in Practice**

Our commitment to transparency manifests in several ways:

**During Extraction**:
- Comprehensive logging of all processing steps
- Warnings logged for low-confidence extractions
- Errors logged with full context for debugging

**In Output**:
- Every populated field has complete provenance
- No "black box" extractions without source attribution
- Confidence scores provided for all values

**For Auditing**:
- Debug mode provides complete decision trail
- Provenance JSON enables independent verification
- Log files capture full execution history

### **Why This Matters for Procurement**

In government procurement, transparency isn't optional—it's a legal and ethical requirement. This system was designed with the understanding that:

1. **Automated decisions must be explainable**: We provide not just answers, but the evidence supporting those answers
2. **Auditability is critical**: Complete provenance enables compliance with procurement regulations
3. **Trust must be earned**: By showing our work, we enable stakeholders to verify accuracy independently
4. **Continuous improvement requires visibility**: Detailed provenance and debugging enable systematic refinement

---

## 🔧 **Technical Implementation**

### **Document Parsing Strategy**

The system employs a sophisticated document parsing approach that handles multiple formats while maintaining precise source tracking:

**PDF Processing**:
- Page-by-page text extraction with PyMuPDF
- Unicode normalization (NFKD) for consistent encoding
- Smart quote and special character handling
- Overlapping chunks (600 chars, 100 char overlap) to preserve context
- Page-level tracking for precise provenance

**HTML Processing**:
- Tag-aware extraction with BeautifulSoup
- Removal of non-content elements (scripts, styles, navigation)
- Text normalization matching PDF processing
- Same chunking strategy for consistency

**Platform Compatibility**:
- Windows-specific PyMuPDF handling to prevent document lifecycle issues
- Proper resource cleanup using finally blocks
- Cross-platform path handling

### **Multi-Query Retrieval**

The retrieval system uses a multi-query approach where each field has multiple semantic queries designed to capture different ways information might be expressed in documents.

**Query Design Philosophy**:
- **Semantic Variety**: Multiple phrasings of the same concept
- **Section Indicators**: Terms that identify where information appears
- **Pattern Recognition**: Format-specific queries for structured data
- **Domain Context**: Industry-specific terminology

**Field Complexity Classification**:
- **Simple fields** (K=5): Single-value fields typically found in one location
- **Medium fields** (K=8): Section-specific data that may span multiple areas
- **Complex fields** (K=12): Table data or information scattered across documents

### **Extraction Strategy**

The system uses a single-pass extraction approach with a carefully engineered prompt that includes:

1. **General Extraction Rules**: Universal guidelines for all fields
2. **Field-Specific Instructions**: Critical rules for problematic fields
3. **Retrieved Context**: All relevant snippets with source labels
4. **Output Format**: Structured examples and schema
5. **Final Reminders**: Emphasis on critical requirements

This approach balances efficiency (single API call) with quality (comprehensive instructions).

### **Validation & Normalization Pipeline**

Post-extraction, the system applies rigorous validation and normalization:

**Validation**:
- Null count checks (alerts if excessive)
- Required field verification
- Product specification completeness
- Cross-field consistency checks

**Normalization**:
- Date formatting (YYYY-MM-DD)
- Phone number formatting (xxx-xxx-xxxx)
- Email normalization (lowercase)
- Array deduplication with order preservation
- Contact information flattening
- Pydantic schema validation

---

## 🎓 **Design Decisions**

### **1. Single-Pass Extraction**

**Decision**: Extract all fields in one LLM call  
**Alternative**: Multiple calls per field  

**Rationale**:
- **Efficiency**: Significantly fewer API calls and faster processing
- **Context**: LLM sees all fields simultaneously, enabling cross-field validation
- **Consistency**: Single prompt ensures consistent extraction logic
- **Cost-Effective**: Reduced token usage

**Trade-off**: Requires more sophisticated prompt engineering

---

### **2. Multi-Query Retrieval**

**Decision**: 4-23 queries per field depending on complexity  
**Alternative**: Single query per field  

**Rationale**:
- **Robustness**: Multiple paths to find the same information
- **Higher Recall**: Increased probability of finding scattered data
- **Generalizability**: Semantic variety works across different document styles
- **Fail-Safe**: If one query fails, others may succeed

---

### **3. Adaptive K Values**

**Decision**: Dynamic K based on field complexity (K=5/8/12)  
**Alternative**: Fixed K for all fields  

**Rationale**:
- **Efficiency**: Low K for simple fields saves tokens and improves speed
- **Completeness**: High K for complex fields (tables, scattered data)
- **Cost Optimization**: Balances quality with resource usage
- **Field-Specific Needs**: Different data types have different retrieval requirements

---

### **4. Multi-Level Provenance**

**Decision**: Primary source + contributing sources + confidence scores  
**Alternative**: Single source attribution only  

**Rationale**:
- **Complete Transparency**: Stakeholders see full information flow
- **Debugging Capability**: Identifies whether issues are retrieval or extraction
- **Audit Trail**: Meets procurement compliance requirements
- **Trust Building**: Enables independent verification by reviewers
- **Quality Improvement**: Detailed provenance guides system refinement

This decision reflects our commitment to transparency over simplicity.

---

### **5. Debug Mode Design**

**Decision**: Optional complete transparency mode  
**Alternative**: Standard logging only  

**Rationale**:
- **Auditability**: Procurement contexts require full decision traceability
- **Verification**: Enables stakeholders to examine extraction reasoning
- **Development**: Facilitates system improvement and debugging
- **Optional**: Doesn't impact standard operation, available when needed

---

### **6. Platform-Specific Optimizations**

**Decision**: Windows-specific PyMuPDF handling  
**Alternative**: Single implementation for all platforms  

**Rationale**:
- **Reliability**: PyMuPDF has platform-specific behaviors
- **Robustness**: Proper cleanup prevents document lifecycle issues
- **User Experience**: System works reliably across deployment environments

---

## ⚠️ **Known Limitations**

### **1. Complex Table Parsing**

Part numbers and data in complex table layouts may require additional retrieval chunks. The system uses high K values and pattern matching to mitigate this, but highly complex multi-column tables may benefit from dedicated table extraction libraries.

**Current Mitigation**: K=12 for table-heavy fields, pattern-based queries

---

### **2. Handwritten or Scanned Documents**

The system expects digital text and does not include OCR capabilities. Documents must be digital PDFs or HTML, not scanned images.

**Current Mitigation**: Preprocessing recommendation for OCR (Tesseract) if needed

---

### **3. Non-Standard Document Formats**

Some agencies use unique document structures or terminology that may require query adaptation.

**Current Mitigation**: Pattern-based queries designed to catch variations

---

### **4. API Dependence**

The system relies on OpenAI API, which introduces considerations for cost, latency, and availability.

**Current Mitigation**: Using cost-effective GPT-4o-mini, error handling, comprehensive logging

---

### **5. Language Support**

Currently optimized for English-language procurement documents.

**Current Mitigation**: Unicode normalization handles special characters and diacritics

---

## 🔮 **Future Enhancements**

### **Short-Term Improvements**

1. **Enhanced Table Extraction**
   - Integration with specialized table parsing libraries
   - Cell-level provenance for tabular data
   - Better handling of multi-column layouts

2. **Additional Document Types**
   - Excel spreadsheets (.xlsx)
   - Word documents (.docx)
   - Email threads (.eml, .msg)

3. **Expanded Validation**
   - Field-specific validation rules
   - Industry-specific compliance checks
   - Automatic anomaly detection

4. **Performance Optimization**
   - Batch processing for multiple RFPs
   - Caching for repeated documents
   - Parallel processing where applicable

---

### **Medium-Term Enhancements**

1. **Active Learning Pipeline**
   - User feedback integration
   - Query refinement based on extraction patterns
   - Confidence threshold tuning

2. **Enhanced Provenance**
   - Sentence-level attribution
   - Extraction confidence explanations
   - Visual source highlighting in UI

3. **Domain Adaptation**
   - Healthcare RFPs (HIPAA compliance)
   - Construction bids
   - IT services (SLA parsing)

4. **Quality Metrics Dashboard**
   - Real-time monitoring
   - Field-level success tracking
   - Document complexity analysis

---

### **Long-Term Vision**

1. **Local Model Support**
   - Fine-tuned open-source models
   - On-premise deployment option
   - Reduced operational costs

2. **Multimodal Processing**
   - Diagram and chart extraction
   - Signature verification
   - Logo and brand detection

3. **Interactive Verification Interface**
   - Side-by-side document viewer
   - Click-to-verify provenance
   - Inline editing with re-extraction

4. **Enterprise Integration**
   - Multi-user collaboration
   - Approval workflows
   - Integration with procurement systems

---

## 🤝 **Contributing**

This project demonstrates advanced RAG techniques for procurement automation. Key areas for contribution:

1. **Query Expansion**: Domain-specific query templates for other industries
2. **Schema Extensions**: Additional field types for specialized RFPs
3. **Validation Rules**: Industry-specific compliance checks
4. **Performance**: Optimization for large-scale processing

---

## 📄 **License**

This project is developed for academic and demonstration purposes. Commercial use requires appropriate licensing considerations for:
- OpenAI API usage terms
- PyMuPDF licensing (AGPL)
- Procurement data sensitivity

---

## 🙏 **Acknowledgments**

- **OpenAI**: GPT-4o-mini and text-embedding-3-small models
- **PyMuPDF**: Robust PDF processing library
- **Pydantic**: Schema validation framework
- **Open Source Community**: BeautifulSoup, NumPy, and supporting libraries

---

## 📊 **Summary**

This RFP extraction system demonstrates professional-grade document processing with:

✅ **Advanced RAG Implementation**: Multi-query retrieval with adaptive strategies  
✅ **Robust Multi-Format Parsing**: Cross-platform PDF and HTML processing  
✅ **Complete Transparency**: Multi-level provenance tracking at every stage  
✅ **Production Quality**: Comprehensive validation, normalization, and error handling  
✅ **Generalizable Design**: Domain-aware patterns without hard-coding  

**Above all, this system prioritizes transparency and auditability.** Every design decision, from multi-level provenance to debug mode, reflects a commitment to building automated systems that stakeholders can trust and verify. In high-stakes procurement contexts, this transparency isn't optional—it's essential.

The system processes procurement documents with confidence in accuracy while maintaining complete traceability. All extracted data can be independently verified against source documents, with full visibility into the extraction process when needed.

---

**Built with transparency. Validated with rigor. Documented with care.** 🎯
