# app.py
import streamlit as st
from rag_pipeline import VectorIndex, Generator, RAGPipeline
from PyPDF2 import PdfReader

# --- Streamlit Config ---
st.set_page_config(page_title="Campus Knowledge Copilot", layout="wide")
st.title("🎓 Campus Knowledge Copilot (RAG)")
st.caption("Upload a PDF, ask questions, and get answers with context.")

# --- Sidebar for PDF upload ---
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Main Logic ---
if uploaded_file is not None:
    with st.spinner("Reading and indexing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]  # Simple chunking

        # Build Vector DB
        vector_index = VectorIndex()
        vector_index.add_texts(chunks)

        # Hugging Face Generator
        generator = Generator()

        # RAG Pipeline
        rag = RAGPipeline(vector_index, generator)

    st.success("✅ Knowledge Base Ready! Ask your questions below.")

    # Question input
    query = st.text_input("💬 Ask a question about the uploaded PDF:")

    if query:
        with st.spinner("Thinking... 🤔"):
            answer = rag.run(query, top_k=5)
            st.write("### 📌 Answer:")
            st.write(answer)

            # Optional: Show retrieved chunks
            with st.expander("📄 Retrieved Chunks (Context)"):
                results = vector_index.query(query, top_k=5)
                for idx, chunk in enumerate(results, start=1):
                    st.markdown(f"**Chunk {idx}:** {chunk[:500]}...")
else:
    st.info("⬅️ Please upload a PDF from the sidebar to get started.")
