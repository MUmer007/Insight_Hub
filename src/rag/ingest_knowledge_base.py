"""
ingest_knowledge_base.py

Phase 4 - GenAI/RAG: Step 1 - Build the searchable knowledge base

What this does:
  1. Read every .md file in data/knowledge_base/
  2. CHUNK: split each document into smaller pieces (models work better
     retrieving small, focused chunks than whole documents)
  3. EMBED: convert each chunk into a vector (a list of numbers) that
     captures its meaning, using a small local model (no API needed)
  4. STORE: save those vectors in ChromaDB, a database built specifically
     for searching by "meaning" instead of exact keyword matches

Run it with:  uv run python src/rag/ingest_knowledge_base.py
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs fine on CPU
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks so we don't cut sentences awkwardly mid-idea


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(directory: Path) -> list[dict]:
    """Read every .md file and return a list of {source, text} dicts."""
    documents = []
    for filepath in directory.glob("*.md"):
        text = filepath.read_text(encoding="utf-8")
        documents.append({"source": filepath.name, "text": text})
    return documents


def main():
    print(f"Loading documents from: {KNOWLEDGE_BASE_DIR}")
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"Found {len(documents)} documents: {[d['source'] for d in documents]}")

    # Chunk every document
    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_counter = 0

    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": doc["source"]})
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    print(f"Created {len(all_chunks)} chunks total.")

    # Load the embedding model (downloads once, then cached locally)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Generating embeddings for all chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    # Store everything in a persistent Chroma database
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # If this collection already exists from a previous run, delete it first
    # so re-running this script doesn't create duplicate entries
    existing = [c.name for c in client.list_collections()]
    if "support_docs" in existing:
        client.delete_collection("support_docs")

    collection = client.create_collection(name="support_docs")
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )

    print(f"\nStored {len(all_chunks)} chunks in ChromaDB at: {CHROMA_DB_DIR}")
    print("Knowledge base ingestion complete.")


if __name__ == "__main__":
    main()
