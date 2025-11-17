import os
import argparse
import requests
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DOCS_URLS = {
    "langgraph": "https://langchain-ai.github.io/langgraph/llms-full.txt",
    "langchain": "https://python.langchain.com/llms.txt"
}

DATA_DIR = "data/raw"
VECTORSTORE_DIR = "data/vectorstore"


def download_docs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in DOCS_URLS.items():
        print(f"Downloading {name} docs...")
        response = requests.get(url)
        filepath = os.path.join(DATA_DIR, f"{name}_docs.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)


def build_vectorstore(force_rebuild=False):
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                documents.append(Document(
                    page_content=f.read(),
                    metadata={"source": filename.replace("_docs.txt", "")}
                ))

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Using HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    batch_size = 100

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    if force_rebuild:
        try:
            client.delete_collection("langgraph_docs")
            print("Deleted existing collection for fresh rebuild")
        except:
            pass
        collection = client.create_collection("langgraph_docs", metadata={"hnsw:space": "cosine"})
        existing_count = 0
    else:
        try:
            collection = client.get_collection("langgraph_docs")
            existing_count = collection.count()
            print(f"Found existing collection with {existing_count} documents, resuming...")
        except:
            collection = client.create_collection("langgraph_docs", metadata={"hnsw:space": "cosine"})
            existing_count = 0

    start_idx = existing_count
    for i in range(start_idx, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}")

        collection.add(
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
            ids=[f"doc_{j}" for j in range(i, i + len(batch))],
            embeddings=embeddings.embed_documents([d.page_content for d in batch])
        )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare LangGraph/LangChain documentation for offline use")
    parser.add_argument("--force-rebuild", action="store_true", help="Force complete rebuild of vector store (deletes existing collection)")
    args = parser.parse_args()

    download_docs()
    build_vectorstore(force_rebuild=args.force_rebuild)
