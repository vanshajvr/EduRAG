# EduRAG: Campus Knowledge Copilot

EduRAG is a Retrieval-Augmented Generation (RAG) application that allows users to **upload PDFs** and **ask questions** about their content. It returns answers along with **source citations**, making it a useful tool for students, researchers, and anyone working with documents.

---

## Features

- Upload PDF files as knowledge bases
- Semantic search using FAISS vector embeddings
- Free, local LLM (`google/flan-t5-small`) for text generation
- Answers include source document snippets
- Streamlit-based web interface for interactive usage

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/LLM-RAG-Project-Campus-Knowledge-Copilot.git
cd LLM-RAG-Project-Campus-Knowledge-Copilot
```
2. **Create a virtual environment and activate it**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Upload a PDF from the sidebar.
3. Wait for the app to process and embed the document.
4. Ask questions about the PDF content and get answers along with source snippets.

---

## Project Structure
```
EduRAG
├── app.py              # Streamlit interface
├── rag_pipeline.py     # RAG pipeline: document processing, vector DB, QA chain
├── requirements.txt    # Project dependencies
└── README.md           # Project description
```
---

## Notes
- This setup uses local free models:
- LLM: google/flan-t5-small
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- No API keys are required; fully free and deployable on Streamlit Cloud.
- Suitable for small to medium PDFs; large documents may require more memory.

---

## License
This project is open-source and available under the MIT License.
