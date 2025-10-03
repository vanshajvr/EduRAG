# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load .env (if needed)
load_dotenv()

# --- Load PDF ---
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# --- Split into chunks ---
def split_documents(documents):
    # Smaller chunks for better retrieval relevance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# --- Create vector DB ---
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- Build QA chain with local Flan-T5 ---
def build_qa_chain(vector_db, top_k=5):
    # Use Flan-T5 small/base for local inference
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=generator)

    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    
    # Map-Reduce gives better aggregation across chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )
    return qa_chain

# --- Ask question ---
def ask_question(qa_chain, query: str):
    result = qa_chain({"query": query})
    return result
