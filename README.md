
````markdown
# RFP-Extractor

Automated information extraction system for procurement/bid documents (RFPs).  
Parses PDF and HTML bid documents, uses OpenAI embeddings + GPT-5 Nano to extract structured information into JSON format with provenance tracking and precedence rules.

## Features

- Supports PDF + HTML input documents  
- Semantic retrieval via OpenAI `text-embedding-3-small` model  
- Single-pass extraction using GPT-5 Nano  
- Precedence: Addendums > Base PDF > HTML  
- Simple provenance tracking (source file & page)  
- Arrays for multiple products, part numbers, models  
- Page-based chunking (no overlap)  
- Detailed logging for debugging (`LOG_LEVEL=DEBUG`)  
- Output JSON per bid in `outputs/` directory

## Getting Started

### Prerequisites

- Python 3.9+  
- An OpenAI API key  

### Setup

1. Clone the repo  
   ```bash
   git clone https://github.com/adiavolo/RFP-Extractor.git
   cd RFP-Extractor
````

2. Create and activate a virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and set:

   ```env
   OPENAI_API_KEY=sk-yourkeyhere
   EMBED_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-5-nano
   LOG_LEVEL=DEBUG
   ```

## Usage

```bash
python rfp_extractor.py \
  --inputs "path/to/base-RFP.pdf" "path/to/notice.html" \
  [optional: additional addendum/spec PDFs] \
  --out outputs \
  --bid_id BID12345 \
  --provenance on
```

**Notes:**

* At least one PDF (base solicitation) and one HTML notice are required.
* Additional files (addendums/specs) may be added to improve extraction.
* The script will write:

  * `outputs/BID12345.json` – main structured JSON output
  * `outputs/BID12345.provenance.json` (if `--provenance on`) – source tracking data
* The script exits with status code `1` if HTML input is missing (as per assignment guideline).

## Directory Structure

```
RFP-Extractor/
├── rfp_extractor.py
├── schema_and_prompts.py
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── outputs/
```

## Logging & Debugging

* To view detailed logs (including chunking, retrieval hits, snippet previews) set `LOG_LEVEL=DEBUG` in `.env`.
* Log file is written to `logs/rfp_extractor.log` and also output to the console.

