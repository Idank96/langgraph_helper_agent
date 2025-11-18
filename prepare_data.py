"""
Data preparation script for LangGraph/LangChain Helper.

Downloads official documentation and builds a ChromaDB vector store for offline retrieval.
"""

import os
import argparse
import requests
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Documentation source URLs
DOCS_URLS = {
    "langgraph": "https://langchain-ai.github.io/langgraph/llms-full.txt",
    "langchain": "https://python.langchain.com/llms.txt"
}

DATA_DIR = "data/raw"
VECTORSTORE_DIR = "data/vectorstore"


def download_docs():
    """Download LangGraph and LangChain documentation from official sources."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, url in DOCS_URLS.items():
        print(f"📥 Downloading {name} documentation...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filepath = os.path.join(DATA_DIR, f"{name}_docs.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)

            # Show file size
            size_kb = len(response.text) / 1024
            print(f"   ✓ Downloaded {name} ({size_kb:.1f} KB) to {filepath}")

        except Exception as e:
            print(f"   ✗ Error downloading {name}: {e}")
            raise

    print()


def build_vectorstore(force_rebuild=False):
    """
    Build ChromaDB vector store from downloaded documentation.

    Args:
        force_rebuild: If True, delete existing collection and rebuild from scratch
    """
    # Load documents
    print("📚 Loading documentation files...")
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename.replace("_docs.txt", "")}
                ))
                print(f"   ✓ Loaded {filename} ({len(content)} chars)")

    # Split into chunks
    print("\n✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(documents)
    print(f"   ✓ Created {len(splits)} chunks (1000 chars each, 200 overlap)")

    # Initialize embeddings
    print("\n🧠 Initializing embeddings model...")
    print("   Model: sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("   ✓ Embeddings model loaded")

    # Initialize ChromaDB
    print(f"\n💾 Setting up ChromaDB at {VECTORSTORE_DIR}...")
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # Handle force rebuild
    if force_rebuild:
        try:
            client.delete_collection("langgraph_docs")
            print("   ✓ Deleted existing collection for fresh rebuild")
        except:
            pass
        collection = client.create_collection(
            "langgraph_docs",
            metadata={"hnsw:space": "cosine"}
        )
        existing_count = 0
    else:
        try:
            collection = client.get_collection("langgraph_docs")
            existing_count = collection.count()
            print(f"   ✓ Found existing collection with {existing_count} documents")
            print("   ℹ️  Will resume from where we left off...")
        except:
            collection = client.create_collection(
                "langgraph_docs",
                metadata={"hnsw:space": "cosine"}
            )
            existing_count = 0
            print("   ✓ Created new collection")

    # Add documents in batches
    print(f"\n⚡ Adding documents to vector store...")
    batch_size = 100
    start_idx = existing_count

    total_batches = (len(splits) - start_idx + batch_size - 1) // batch_size

    for i in range(start_idx, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        batch_num = (i - start_idx) // batch_size + 1

        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} documents)...", end=" ")

        try:
            collection.add(
                documents=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
                ids=[f"doc_{j}" for j in range(i, i + len(batch))],
                embeddings=embeddings.embed_documents([d.page_content for d in batch])
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise

    # Final count
    final_count = collection.count()
    print(f"\n✅ Done! Vector store contains {final_count} documents")
    print(f"   Location: {VECTORSTORE_DIR}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare LangGraph/LangChain documentation for offline use"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force complete rebuild of vector store (deletes existing collection)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" LangGraph/LangChain Documentation Preparation")
    print("=" * 70)
    print()

    try:
        download_docs()
        build_vectorstore(force_rebuild=args.force_rebuild)

        print("=" * 70)
        print(" Setup Complete!")
        print("=" * 70)
        print("\nYou can now run the agent:")
        print('  python main.py "How do I use StateGraph?"')
        print()

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        exit(1)
