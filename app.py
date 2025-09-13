# app.py
import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vector_db, build_qa_chain, ask_question

# --- Streamlit page config ---
st.set_page_config(page_title="EduRAG", layout="wide")
st.title("EduRAG")
st.markdown("Upload PDFs and ask questions. Answers come with source citations.")

# --- Sidebar: PDF upload ---
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# --- Sidebar: settings ---
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K documents to retrieve", min_value=1, max_value=10, value=3)

# --- Process uploaded PDF ---
if uploaded_file:
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF..."):
        # Load, split, and create vector DB
        documents = load_documents(pdf_path)
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)
        qa_chain = build_qa_chain(vector_db)

    st.success("Knowledge base ready! Ask your questions below.")

    # --- Ask questions ---
    query = st.text_input("Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Thinking..."):
            answer_data = ask_question(qa_chain, query)
            st.subheader("Answer:")
            st.write(answer_data["result"])

            # Show source docs
            if "source_documents" in answer_data:
                with st.expander("Source Documents"):
                    for doc in answer_data["source_documents"]:
                        st.markdown(doc.page_content[:500] + "...")

else:
    st.info("Please upload a PDF from the sidebar to get started.")
