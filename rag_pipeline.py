# rag_pipeline.py (Enhanced Semantic RAG + Re-ranking)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import CrossEncoder
import numpy as np

# Load environment variables (optional)
load_dotenv()

# --- Load PDF ---
def load_documents(pdf_path: str):
    return PyPDFLoader(pdf_path).load()

# --- Split into chunks ---
def split_documents(documents, chunk_size=600, overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)

# --- Create vector DB ---
def create_vector_db(chunks):
    # Better semantic embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- Re-rank retrieved docs ---
def rerank_documents(query, docs, top_n=3):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    ranked = [docs[i] for i in np.argsort(scores)[::-1][:top_n]]
    return ranked

# --- Build QA Chain ---
def build_qa_chain(vector_db, top_k=5):
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=generator)

    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )
    return qa_chain

# --- Ask Question (with re-ranking) ---
def ask_question(qa_chain, query: str):
    result = qa_chain.invoke({"query": query})
    docs = result["source_documents"]

    # Re-rank before final answer generation
    ranked_docs = rerank_documents(query, docs)
    combined_text = " ".join([d.page_content for d in ranked_docs])

    refined_answer = qa_chain.llm(f"Answer this question based only on the text:\n\n{combined_text}\n\nQ: {query}")
    return {
        "result": refined_answer,
        "source_documents": ranked_docs
    }
