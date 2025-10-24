
# 🧩 RFP Information Extractor

A lightweight NLP + LLM pipeline that automatically extracts structured information from **Request for Proposal (RFP)** documents (PDF + HTML).
Uses **OpenAI embeddings** for semantic retrieval and **GPT-5 Nano** for context-aware field extraction into a standardized JSON format.

---

## 🚀 Features

* Parses **PDF** and **HTML** bid documents
* Performs **semantic chunking** and **embedding-based retrieval**
* Runs **LLM extraction** with precedence handling (Addendum > Base PDF > HTML)
* Outputs clean, validated **JSON** and **provenance** files
* Detailed **debug logs** for traceability

---

## 🧱 Project Structure

```
rfp_extractor/
├── rfp_extractor.py          # Main driver script
├── schema_and_prompts.py     # Schema, field cues, and prompt templates
├── .env                      # Environment variables (API keys etc.)
├── .gitignore
├── requirements.txt
├── README.md
└── outputs/                  # JSON and provenance files generated here
```

---

## ⚙️ Setup

### 1. Clone & enter

```bash
git clone https://github.com/<yourusername>/rfp_extractor.git
cd rfp_extractor
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the root folder:

```bash
OPENAI_API_KEY=sk-yourkeyhere
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-5-nano
LOG_LEVEL=DEBUG
```

---

## 🧠 Usage

Example command:

```bash
python rfp_extractor.py \
  --inputs "bids/Bid2/PORFP_-_Dell_Laptop_Final.pdf" \
           "bids/Bid2/Dell_Laptops_Bid_Info.html" \
  --out outputs \
  --bid_id E20P4600040 \
  --provenance on
```

---

## 🧾 Output

* `outputs/<bid_id>.json` → structured RFP information
* `outputs/<bid_id>.provenance.json` → source tracing (file + page)

Example summary after run:

```
============================================================
Extraction Summary for Bid E20P4600040
============================================================
Fields populated: 18/21
Products found: 2
============================================================
```

---

## 🧰 Debugging

* Logs printed to console **and** saved in `logs/rfp_extractor.log`
* Adjust verbosity via `.env` → `LOG_LEVEL=DEBUG | INFO`

---

## 📜 License

MIT License © 2025 Adithya Sai
