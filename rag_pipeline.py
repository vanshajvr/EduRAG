import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint


# --- Load .env ---
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY", "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")  # fallback to your key


# --- Load documents ---
def load_documents(pdf_path: str):
    """Load PDF and return documents"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# --- Split documents ---
def split_documents(documents):
    """Split into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# --- Create vector DB ---
def create_vector_db(chunks):
    """Create FAISS vector DB with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# --- Build QA chain ---
def build_qa_chain(vector_db):
    """Build RetrievalQA chain using HuggingFace Inference API"""
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # free instruct-tuned model
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
    """Run query on RAG pipeline"""
    result = qa_chain.invoke({"query": query})
    return result
