# app.py
import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vector_db, build_pipeline, ask_question
from graphviz import Digraph
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import tempfile

st.set_page_config(page_title="EduRAG", layout="wide")
st.title("EduRAG – Campus Knowledge Copilot")
st.markdown("Upload PDFs and ask questions. Answers come with source citations.")

# Sidebar
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K documents to retrieve", 1, 10, 3)
chunk_size = st.sidebar.slider("Chunk size (characters)", 400, 1200, 600, step=100)
overlap = st.sidebar.slider("Chunk overlap (characters)", 50, 300, 120, step=10)
rerank_enabled = st.sidebar.checkbox("Enable cross-encoder re-ranking (may be slower)", value=True)
rerank_top_n = st.sidebar.slider("Re-rank top N", 1, 5, 3)

# Process upload
if uploaded_file:
    # Save to temp file to avoid permission issues
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        pdf_path = tmp.name

    st.sidebar.success("PDF uploaded")

    # Load docs
    try:
        documents = load_documents(pdf_path)
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        st.stop()

    if not documents:
        st.error("No content extracted from PDF. Try a different file.")
        st.stop()

    st.info(f"Loaded {len(documents)} document objects. Splitting into chunks...")

    # Split
    try:
        chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
    except Exception as e:
        st.error(f"Error while splitting documents: {e}")
        st.stop()

    st.info(f"Created {len(chunks)} chunks. Building vector index...")

    # Create vector DB (embedding model can be changed)
    try:
        vector_db = create_vector_db(chunks, embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    except Exception as e:
        st.error(f"Failed to create vector DB: {e}")
        st.stop()

    st.info("Vector DB ready. Building retriever & generator (this may download model on first run)...")

    # Build pipeline (retriever + generator)
    try:
        retriever, generator = build_pipeline(vector_db, llm_model=os.getenv("LLM_MODEL", "google/flan-t5-small"), top_k=top_k)
    except Exception as e:
        st.error(f"Failed to build pipeline: {e}")
        st.stop()

    st.success("Knowledge base ready! Ask your question below.")

    query = st.text_input("Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Retrieving & generating answer..."):
            try:
                result = ask_question(retriever, generator, query, rerank=rerank_enabled, rerank_top_n=rerank_top_n)
            except Exception as e:
                st.error(f"Error during QA: {e}")
                result = {"result": f"Error: {e}", "source_documents": []}

            st.subheader("Answer:")
            st.write(result.get("result", "No answer returned."))

            if result.get("source_documents"):
                st.subheader("Source snippets (top results):")
                for i, d in enumerate(result["source_documents"], 1):
                    src = d.metadata.get("source", "Unknown")
                    snippet = d.page_content[:600].replace("\n", " ")
                    st.markdown(f"**{i}. {src}**")
                    st.write(snippet)
    # Flowchart + mindmap (unchanged)
    st.subheader("Pipeline Flowchart")
    dot = Digraph(comment="RAG Pipeline")
    dot.node('A', 'Upload PDF'); dot.node('B', 'Load & Split Docs'); dot.node('C', 'Embed Chunks -> FAISS')
    dot.node('D', 'Retrieve Top-K'); dot.node('E', 'Re-rank (optional)'); dot.node('F', 'LLM generates answer')
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])
    st.graphviz_chart(dot)

    st.subheader("Interactive Mindmap")
    net = Network(height="350px", width="100%", notebook=True)
    nodes = ['Upload PDF', 'Load & Split Docs', 'FAISS DB', 'Retrieve Top-K', 'Re-rank', 'LLM Answer']
    for n in nodes:
        net.add_node(n, label=n)
    edges = [('Upload PDF','Load & Split Docs'),('Load & Split Docs','FAISS DB'),('FAISS DB','Retrieve Top-K'),
             ('Retrieve Top-K','Re-rank'),('Re-rank','LLM Answer')]
    for e in edges:
        net.add_edge(*e)
    net.save_graph("rag_mindmap.html")
    with open("rag_mindmap.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=450)

else:
    st.info("Please upload a PDF from the sidebar to get started.")
