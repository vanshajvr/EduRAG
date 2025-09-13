import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vector_db, build_qa_chain, ask_question
from graphviz import Digraph
from pyvis.network import Network
import streamlit.components.v1 as components

# Streamlit page config
st.set_page_config(page_title="EduRAG", layout="wide")
st.title("EduRAG – Campus Knowledge Copilot")
st.markdown("Upload PDFs and ask questions. Answers come with source citations.")

# Sidebar: Upload PDF
st.sidebar.header("Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Sidebar: Settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K documents to retrieve", 1, 10, 3)

# HF API Key
hf_api_key = st.secrets.get("HF_API_KEY", "")
if not hf_api_key:
    st.warning("HF_API_KEY not found! Add it in Streamlit secrets to use the app.")
    st.stop()

# Process uploaded PDF
if uploaded_file:
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("PDF uploaded successfully!")

    # Load documents
    with st.spinner("Processing PDF..."):
        documents = load_documents(pdf_path)
        chunks = split_documents(documents)
        vector_db = create_vector_db(chunks)
        qa_chain = build_qa_chain(vector_db)
    st.success("Knowledge base ready! Ask your questions below.")

    # Ask questions
    query = st.text_input("Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Thinking..."):
            answer_data = ask_question(qa_chain, query)
            st.subheader("Answer:")
            st.write(answer_data["result"])

            st.subheader("Sources:")
            for doc in answer_data["source_documents"]:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")

    # --- Visual Flowchart ---
    st.subheader("Pipeline Flowchart")
    dot = Digraph(comment="RAG Pipeline")
    dot.node('A', 'Upload PDF')
    dot.node('B', 'Load & Split Docs')
    dot.node('C', 'Embed Chunks → FAISS DB')
    dot.node('D', 'User Enters Question')
    dot.node('E', 'Retrieve Top-K Chunks')
    dot.node('F', 'LLM Generates Answer')
    dot.node('G', 'Return Answer + Sources')
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])
    st.graphviz_chart(dot)

    # --- Interactive Mindmap ---
    st.subheader("Interactive Mindmap")
    net = Network(height="400px", width="100%", notebook=True)
    nodes = ['Upload PDF', 'Load & Split Docs', 'FAISS DB', 'User Question', 'Retrieve Chunks', 'LLM Answer', 'Return Answer']
    for node in nodes:
        net.add_node(node, label=node)
    edges = [('Upload PDF', 'Load & Split Docs'),
             ('Load & Split Docs', 'FAISS DB'),
             ('FAISS DB', 'Retrieve Chunks'),
             ('User Question', 'Retrieve Chunks'),
             ('Retrieve Chunks', 'LLM Answer'),
             ('LLM Answer', 'Return Answer')]
    for edge in edges:
        net.add_edge(*edge)
    net.save_graph("rag_mindmap.html")
    HtmlFile = open("rag_mindmap.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=500)

else:
    st.info("Please upload a PDF from the sidebar to get started.")
