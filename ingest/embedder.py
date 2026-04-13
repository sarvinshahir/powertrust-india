"""
PowerTrust India Solar Intelligence
Embedder — chunks documents and builds ChromaDB vector store
Usage: python ingest/embedder.py
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARSED_FILE = "data/parsed_docs.json"
CHROMA_DIR = "data/chroma_db_local"
COLLECTION_NAME = "india_solar_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def smart_chunk(text):
    """
    Adaptive chunking based on document length:
    - Short docs (<5000 chars): keep whole
    - Medium docs (<20000 chars): 1500 char chunks, 200 overlap
    - Large docs (>20000 chars): 2500 char chunks, 400 overlap
    Breaks at paragraph or sentence boundaries where possible.
    """
    text = text.strip()
    doc_len = len(text)

    if doc_len < 5000:
        return [text]

    chunk_size = 1500 if doc_len < 20000 else 2500
    overlap = 200 if doc_len < 20000 else 400

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            break

        # Try paragraph boundary first
        para_break = text.rfind("\n\n", start, end)
        if para_break > start + chunk_size // 2:
            end = para_break
        else:
            # Fall back to sentence boundary
            sent_break = text.rfind(". ", start, end)
            if sent_break > start + chunk_size // 2:
                end = sent_break + 1

        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

        start = end - overlap

    return chunks

def build_vector_store():
    # Load parsed docs
    with open(PARSED_FILE, "r") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents")

    # Setup ChromaDB
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Chunk all documents
    all_chunks, all_ids, all_meta = [], [], []

    for doc in tqdm(docs, desc="Chunking"):
        chunks = smart_chunk(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['filename']}_{i}")
            all_meta.append({
                "filename": doc["filename"],
                "dimension": doc["dimension"],
                "chunk_index": i,
                "doc_length": len(doc["text"])
            })

    # Print stats
    doc_len = [len(d["text"]) for d in docs]
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunk distribution:")
    print(f"  Short docs kept whole (<5k chars): {sum(1 for d in docs if len(d['text']) < 5000)}")
    print(f"  Medium docs (1500 char chunks): {sum(1 for d in docs if 5000 <= len(d['text']) < 20000)}")
    print(f"  Large docs (2500 char chunks): {sum(1 for d in docs if len(d['text']) >= 20000)}")

    # Embed in batches
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_meta[i:i+batch_size]
        )

    print(f"\n✅ Done! {collection.count()} chunks stored in {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

if __name__ == "__main__":
    build_vector_store()
