import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = "data/vectorstore"


def retrieve_context(question: str, k: int = 10) -> str:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_collection("langgraph_docs")

    results = collection.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=k
    )
    return "\n\n".join(results["documents"][0])

