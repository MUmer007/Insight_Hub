"""
rag_query.py

Phase 4 - GenAI/RAG: Step 2 - Answer questions using retrieval + generation

What "RAG" (Retrieval-Augmented Generation) actually means:
  1. RETRIEVE: given a question, find the most relevant chunks from our
     knowledge base (using the embeddings we built in ingest_knowledge_base.py)
  2. AUGMENT: stuff those chunks into the prompt as context
  3. GENERATE: ask the LLM to answer USING that context, not its own
     general knowledge — this is what keeps it grounded in OUR company's
     actual policies instead of making things up

Run it with:  uv run python src/rag/rag_query.py
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# IMPORTANT: check Groq's console (console.groq.com) for currently available
# model names if this one errors — providers update their model lineup over time.
GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 3  # how many chunks to retrieve per question

load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """You are a helpful customer support assistant for an e-commerce company.
Answer the customer's question using ONLY the information in the provided context below.
If the context doesn't contain enough information to answer confidently, say so honestly
and suggest the customer contact human support — do NOT make up policy details.
Keep answers concise and friendly."""


class RagPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = client.get_collection(name="support_docs")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Make sure it's set in your .env file."
            )
        self.groq_client = Groq(api_key=api_key)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Find the most relevant chunks for a given question."""
        query_embedding = self.embedder.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        chunks = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"text": doc, "source": metadata["source"], "distance": distance})
        return chunks

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        """Ask Groq to answer, grounded in the retrieved chunks."""
        context = "\n\n".join(
            f"[From {c['source']}]\n{c['text']}" for c in chunks
        )

        user_message = f"""Context from our knowledge base:
{context}

Customer question: {question}"""

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # lower = more focused/factual, less creative
        )
        return response.choices[0].message.content

    def answer(self, question: str) -> dict:
        """Full RAG pipeline: retrieve, then generate."""
        chunks = self.retrieve(question)
        answer_text = self.generate_answer(question, chunks)
        return {
            "question": question,
            "answer": answer_text,
            "sources": list(set(c["source"] for c in chunks)),
        }


def main():
    rag = RagPipeline()

    # A handful of test questions to sanity-check the pipeline
    test_questions = [
        "How long do I have to return something?",
        "Do you ship internationally?",
        "Why was I charged twice?",
        "What payment methods do you accept?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = rag.answer(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
