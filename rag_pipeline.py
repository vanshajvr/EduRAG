# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

# Load .env (optional, in case running locally)
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY", "")

# --- Load PDF documents ---
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# --- Split documents into chunks ---
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# --- Create FAISS vector database ---
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- Build RAG QA chain ---
def build_qa_chain(vector_db):
    # Use a free Hugging Face inference-ready model
    llm = HuggingFaceEndpoint(
        repo_id="tiiuae/falcon-7b-instruct",
        token=hf_api_key,
        temperature=0.3,
        max_new_tokens=300,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# --- Ask question ---
def ask_question(qa_chain, query: str):
    result = qa_chain({"query": query})
    return result  # result["result"] is the answer, result["source_documents"] has sources

