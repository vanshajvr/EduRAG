# rag_pipeline.py (semantic version)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()

def load_documents(pdf_path: str):
    return PyPDFLoader(pdf_path).load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    return splitter.split_documents(documents)

def create_vector_db(chunks):
    # More semantic embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def build_qa_chain(vector_db, top_k=5):
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # more capable reasoning
        tokenizer="google/flan-t5-base",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=generator)

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )
    return qa_chain

def ask_question(qa_chain, query: str):
    result = qa_chain.invoke({"query": query})
    return result
