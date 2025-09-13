import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()  # Load env vars from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")


def load_documents(pdf_path: str):
    """Load PDF and return documents"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    """Split into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_vector_db(chunks):
    """Create FAISS vector DB"""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


def build_qa_chain(vector_db):
    """Build RetrievalQA chain"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain


def ask_question(qa_chain, query: str):
    """Run query on RAG pipeline"""
    result = qa_chain.invoke({"query": query})
    return result
