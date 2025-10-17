# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import numpy as np

# Lazy import for CrossEncoder (heavy). Import inside function to avoid startup OOMs.
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except Exception:
    CrossEncoder = None
    _HAS_CROSS_ENCODER = False

load_dotenv()


def load_documents(pdf_path: str):
    """Load PDF and return list of LangChain Document objects."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs


def split_documents(documents, chunk_size: int = 600, overlap: int = 120):
    """Split LangChain documents into chunks. Returns list of Documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_db(chunks, embed_model: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Create an in-memory FAISS vector DB from chunks using HuggingFaceEmbeddings.
    Returns the vectorstore instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


def rerank_documents(query: str, docs, top_n: int = 3, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Re-rank retrieved docs using a CrossEncoder. If CrossEncoder is not available,
    this function returns the original docs[:top_n].
    """
    if not _HAS_CROSS_ENCODER:
        # fallback: return first top_n
        return docs[:top_n]

    try:
        model = CrossEncoder(model_name)
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        order = np.argsort(scores)[::-1][:top_n]
        ranked = [docs[i] for i in order]
        return ranked
    except Exception:
        # If reranker fails for any reason, return top_n from original
        return docs[:top_n]


def build_pipeline(vector_db, llm_model: str = "google/flan-t5-small", top_k: int = 5):
    """
    Build the retrieval pipeline:
      - retriever (from FAISS)
      - generator (transformers pipeline)
    Returns (retriever, generator_callable).
    generator_callable(prompt) -> string
    """
    # Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    # Build local generator lazily
    # Use text2text-generation pipeline. This will download model on first call.
    gen = pipeline(
        "text2text-generation",
        model=llm_model,
        tokenizer=llm_model,
        truncation=True,
        max_new_tokens=256,
        device=0 if (os.getenv("USE_GPU", "0") == "1") else -1,
    )

    def generator(prompt: str):
        # Call transformers pipeline and extract text reliably
        out = gen(prompt, max_new_tokens=256, return_full_text=False)
        # transformers pipeline returns a list of dicts: [{'generated_text': '...'}]
        if isinstance(out, list) and len(out) > 0:
            text = out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
            return text
        return str(out)

    return retriever, generator


def ask_question(retriever, generator, query: str, rerank: bool = True, rerank_top_n: int = 3):
    """
    High-level ask:
      - retrieve docs via retriever.get_relevant_documents(query)
      - optionally rerank with CrossEncoder
      - build context and call generator(prompt)
      - return dict: { "result": str, "source_documents": [docs] }
    """
    # Validate
    if retriever is None:
        raise ValueError("Retriever is None. Build pipeline first.")

    docs = retriever.get_relevant_documents(query)

    if not docs:
        return {"result": "No relevant content found in the uploaded documents.", "source_documents": []}

    # Optionally rerank
    if rerank:
        try:
            ranked_docs = rerank_documents(query, docs, top_n=rerank_top_n)
        except Exception:
            ranked_docs = docs[:rerank_top_n]
    else:
        ranked_docs = docs[:rerank_top_n]

    # Build a prompt that instructs the model to answer only from context
    context = "\n\n---\n\n".join([d.page_content for d in ranked_docs])
    prompt = (
        "Answer the question using ONLY the provided context below. "
        "If the answer is not contained in the context, say 'Answer not found in the provided documents.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )

    try:
        answer = generator(prompt)
    except Exception as e:
        # safe fallback
        answer = f"Error generating answer: {e}"

    return {"result": answer, "source_documents": ranked_docs}
