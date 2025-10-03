import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load .env (optional)
load_dotenv()

# --- Load PDF ---
def load_documents(pdf_path: str):
    """Load PDF into document objects"""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# --- Split documents into chunks ---
def split_documents(documents, chunk_size=800, overlap=150):
    """Split text into smaller overlapping chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_documents(documents)

# --- Create vector database ---
def create_vector_db(chunks):
    """Create FAISS vector DB using HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- Build Retrieval QA chain ---
def build_qa_chain(vector_db, top_k=3):
    """
    Builds a RAG chain using a local LLM (Flan-T5 small).
    Uses top_k retrieval to select relevant chunks for context.
    """
    # Local LLM pipeline (text2text)
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=generator)

    # Retriever setup
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    # Retrieval QA with "refine" chain type for better coherence
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="refine",  # instead of "stuff" for better reasoning
        return_source_documents=True
    )
    return qa_chain

# --- Ask question ---
def ask_question(qa_chain, query: str):
    """
    Query the RAG pipeline and return:
    - result: the LLM-generated answer
    - source_documents: list of document chunks used to answer
    """
    result = qa_chain({"query": query})
    return result
