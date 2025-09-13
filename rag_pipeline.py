# rag_pipeline.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Load documents ---
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# --- Split documents ---
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# --- Create vector DB ---
def create_vector_db(chunks):
    embeddings = "sentence-transformers/all-MiniLM-L6-v2"  # small free embedding model
    from langchain_huggingface import HuggingFaceEmbeddings
    emb_model = HuggingFaceEmbeddings(model_name=embeddings)
    vector_db = FAISS.from_documents(chunks, emb_model)
    return vector_db

# --- Build QA chain ---
def build_qa_chain(vector_db):
    # Load small text generation model locally
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)

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
    return result
