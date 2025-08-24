import os
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


from rag_pipeline import VectorIndex, Generator, RAGPipeline


st.set_page_config(page_title="Campus Knowledge Copilot", page_icon="🎓")


st.title("🎓 Campus Knowledge Copilot")
st.caption("Ask questions over your notes, PDFs, and bookmarked URLs. RAG-powered, with source citations.")


# Sidebar config
with st.sidebar:
st.header("Settings")
k = st.slider("Top-K documents", 2, 10, 5)
model = st.text_input("OpenAI Chat Model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
st.divider()
st.markdown
