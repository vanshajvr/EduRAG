# EduRAG

EduRAG is an AI-powered **Retrieval-Augmented Generation (RAG) application** built with **Streamlit** and **LangChain**.  
It allows students and faculty to upload PDFs (syllabus, research papers, notes, handbooks, etc.) and query them conversationally to get **instant, contextual answers** with sources.  

---

## Features
- Upload **any PDF** (books, notes, guidelines).  
- Intelligent **document chunking & vector search** with FAISS.  
- Query using **state-of-the-art LLMs** (OpenAI or Hugging Face).  
- View **answers with sources** for trust & transparency.  
- Simple, interactive **Streamlit UI**.  

---

## Project Structure
```
EduRAG
┣ app.py # Main Streamlit app
┣ rag_pipeline.py # RAG pipeline (load, split, embed, retrieve, QA)
┣ requirements.txt # Dependencies
┣ .env.example # Example environment variables
┗ README.md # Project docs
```

---

## Installation

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/EduRAG.git
cd EduRAG
```
2. **Create a virtual environment & activate it**
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
3.Install dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Variables
- Create a .env file in the project root:
```bash
# For HuggingFace
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```
- Run the App
```bash
streamlit run app.py
```
---

## Usage
1. Upload your PDF file from the sidebar.
2. Wait for the app to process & build knowledge base.
3. Type your question in the text box.
4. Get instant answers with cited sources.

---

## Tech Stack
- Python 3.10+
- Streamlit – frontend
- LangChain – RAG framework
- FAISS – vector search database
- OpenAI / Hugging Face – LLMs
- Sentence-Transformers – embeddings

---

## Example Use Cases
- Students: Quickly query notes, textbooks, or lecture PDFs.
- Researchers: Extract insights from long papers.
- Faculty: Search guidelines and academic policies.

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## License
MIT License
