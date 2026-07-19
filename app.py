import json
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Loading Retrieval API")
print("=" * 60)

EMBEDDING_DIR = Path("embeddings")

SIMILARITY_THRESHOLD = 0.45

# ---------------------------------------------------------
# Load FAISS
# ---------------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(
    str(EMBEDDING_DIR / "corpus.faiss")
)

print(f"FAISS vectors : {index.ntotal}")

# ---------------------------------------------------------
# Load reconstructed chunk metadata
# ---------------------------------------------------------

print("Loading chunk metadata...")

file_path = EMBEDDING_DIR / "chunk_text.json"

print("=" * 50)
print("File exists:", file_path.exists())
print("File path:", file_path)

if file_path.exists():
    print("File size:", file_path.stat().st_size)

    with open(file_path, "r", encoding="utf-8") as f:
        first_300 = f.read(300)

    print("First 300 characters:")
    print(first_300)
print("=" * 50)

with open(file_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)["chunks"]

print(f"Metadata entries : {len(metadata)}")

# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("API Ready.")

# ---------------------------------------------------------

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/retrieve")
def retrieve(req: QueryRequest):

    embedding = model.encode(
        req.query,
        normalize_embeddings=True
    )

    embedding = np.asarray(
        [embedding],
        dtype=np.float32
    )

    scores, indices = index.search(
        embedding,
        req.top_k
    )

    contexts = []
    confidences = []

    for score, idx in zip(scores[0], indices[0]):

        idx = int(idx)

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        meta = metadata[idx]

        text = meta.get("text", "").strip()

        if len(text) == 0:
            continue

        contexts.append({

            "company": meta.get("company"),

            "pdf": meta.get("pdf_file"),

            "pdf_path": meta.get("pdf_path"),

            "chunk": meta.get("chunk_index"),

            "text": text

        })

        confidences.append(float(score))

    # -------------------------------------------------
    # No valid context found
    # -------------------------------------------------

    if len(contexts) == 0:

        return {

            "answer":
            "Not able to find relevant documents. No details are present in the knowledge base.",

            "context": [],

            "confidence": 0.0,

            "num_chunks": 0

        }

    confidence = float(np.mean(confidences))

    return {

        "answer": None,

        "context": contexts,

        "confidence": confidence,

        "num_chunks": len(contexts)

    }