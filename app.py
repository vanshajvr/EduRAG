import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vector_db, build_qa_chain, ask_question

st.set_page_config(page_title="Campus Knowledge Copilot", layout="wide")

st.title("🎓 Campus Knowledge Copilot (RAG)")

# Sidebar for PDF upload
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("✅ PDF uploaded successfully")

    # Load pipeline
    docs = load_documents("uploaded_file.pdf")
    chunks = split_documents(docs)
    vector_db = create_vector_db(chunks)
    qa_chain = build_qa_chain(vector_db)

    st.success("Knowledge Base Ready ✅ Ask your questions below!")

    query = st.text_input("💬 Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Thinking... 🤔"):
            result = ask_question(qa_chain, query)
            st.write("### 📌 Answer:")
            st.write(result["result"])

            # Show sources
            with st.expander("📄 Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content[:500] + "...")
else:
    st.info("Please upload a PDF from the sidebar to get started.")
