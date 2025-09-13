# app.py
import streamlit as st
import os
from rag_pipeline import load_documents, split_documents, create_vector_db, build_qa_chain, ask_question

# Streamlit page config
st.set_page_config(page_title="Campus Knowledge Copilot", layout="wide")
st.title("Campus Knowledge Copilot (RAG)")
st.markdown("Upload PDFs and ask questions. Answers come with source citations.")

# Sidebar: Upload PDFs
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Optional settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K documents to retrieve", min_value=1, max_value=10, value=3)

# Load Hugging Face API key from Streamlit secrets
hf_api_key = st.secrets.get("HF_API_KEY", "")

if not hf_api_key:
    st.warning("HF_API_KEY not found! Add it in Streamlit secrets to use the app.")
    st.stop()

# Process uploaded PDF
if uploaded_file is not None:
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDF uploaded successfully!")

    # Load & split documents
    with st.spinner("Processing PDF..."):
        documents = load_documents(pdf_path)
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)
        qa_chain = build_qa_chain(vector_db)

    st.success("Knowledge base ready! Ask your questions below.")

    # User query
    query = st.text_input("Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Thinking..."):
            result = ask_question(qa_chain, query)
            st.subheader("Answer:")
            st.write(result)

            # Optional: show source documents
            if hasattr(result, "source_documents"):
                with st.expander("Source Documents"):
                    for doc in result.source_documents:
                        st.markdown(f"- {doc.metadata.get('source', 'Unknown source')}: {doc.page_content[:500]}...")
else:
    st.info("Please upload a PDF from the sidebar to get started.")
