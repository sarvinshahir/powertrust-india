"""
PowerTrust India Solar Intelligence
PDF Parser — extracts text from all PDFs and txt files in raw_docs folder
Usage: python ingest/pdf_parser.py
"""

import pdfplumber
import os
import json
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_DIR = "data/raw_docs/pdfs/"
OUTPUT_FILE = "data/parsed_docs.json"

DIMENSION_KEYWORDS = {
    "cost_economics": ["capex", "tariff", "cost", "lcoe", "benchmark", "price",
                       "auction", "ppa", "o&m", "capital", "lakh", "kWh"],
    "grid_access": ["grid", "transmission", "interconnection", "congestion",
                    "open access", "cea", "posoco", "queue", "ists", "gna"],
    "subsidies_policy": ["subsidy", "incentive", "net metering", "pm surya",
                         "kusum", "cfa", "grant", "scheme", "muft bijli"],
    "utility_standards": ["rpo", "renewable purchase", "obligation", "rec",
                          "mandate", "utility", "merc", "cerc"],
    "public_approvals": ["eia", "public hearing", "ngt", "environmental",
                         "clearance", "objection", "approval", "seiaa"],
    "unknown_risks": ["land", "waste", "financing", "barrier", "risk",
                      "delay", "bottleneck", "conflict", "almm", "gst"]
}

def detect_dimension(text):
    text_lower = text.lower()
    scores = {}
    for dim, keywords in DIMENSION_KEYWORDS.items():
        scores[dim] = sum(1 for kw in keywords if kw in text_lower)
    return max(scores, key=scores.get)

def parse_pdf(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Error: {e}")
    return text.strip()

def parse_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def parse_all_docs():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_docs = []
    skipped = []

    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf") or f.endswith(".txt")]
    print(f"Found {len(files)} files ({sum(1 for f in files if f.endswith('.pdf'))} PDFs, {sum(1 for f in files if f.endswith('.txt'))} TXT)")

    for filename in tqdm(files):
        filepath = os.path.join(PDF_DIR, filename)

        if filename.endswith(".pdf"):
            text = parse_pdf(filepath)
        else:
            text = parse_txt(filepath)

        if len(text) < 100:
            print(f"\n  ⚠️  Low text: {filename} — may be scanned PDF")
            skipped.append(filename)
            continue

        dimension = detect_dimension(text)
        all_docs.append({
            "filename": filename,
            "filepath": filepath,
            "dimension": dimension,
            "text": text,
            "char_count": len(text)
        })

    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["filename"] not in seen:
            seen.add(doc["filename"])
            unique_docs.append(doc)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_docs, f, indent=2)

    print(f"\n✅ Parsed {len(unique_docs)} documents → {OUTPUT_FILE}")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} files (scanned/empty): {skipped}")

    from collections import Counter
    dims = Counter(d["dimension"] for d in unique_docs)
    print("\nDimension breakdown:")
    for dim, count in sorted(dims.items(), key=lambda x: -x[1]):
        print(f"  {dim}: {count} docs")

if __name__ == "__main__":
    parse_all_docs()
