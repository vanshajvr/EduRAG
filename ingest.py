# ingest.py
import os
import argparse
from typing import List
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader, PyPDFLoader
from rag_pipeline import VectorIndex

SUPPORTED_EXT = [".txt", ".md"]  # You can expand this list


def load_path(path: str):
    """Load documents from file, folder, or URL"""
    docs = []

    if path.startswith("http://") or path.startswith("https://"):
        loader = UnstructuredURLLoader(urls=[path])
        docs.extend(loader.load())

    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                full = os.path.join(root, fn)

                if ext in SUPPORTED_EXT:
                    docs.extend(TextLoader(full, encoding="utf-8").load())
                elif ext == ".pdf":
                    docs.extend(PyPDFLoader(full).load())

    else:  # Single file
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_EXT:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        else:
            raise ValueError(f"Unsupported file: {path}")

    # Attach metadata
    for d in docs:
        if "source" not in d.metadata:
            d.metadata["source"] = path

    return docs


def main(paths: List[str]):
    all_docs = []
    for p in paths:
        print(f"Loading {p}...")
        all_docs.extend(load_path(p))

    if not all_docs:
        print("⚠️ No documents found.")
        return

    # Build FAISS index
    index_dir = os.getenv("INDEX_DIR", "data/index")
    os.makedirs(index_dir, exist_ok=True)

    index = VectorIndex()

    faiss_file = os.path.join(index_dir, "index.faiss")
    if os.path.isdir(index_dir) and os.path.isfile(faiss_file):
        print("Adding to existing index...")
        index.add([d.page_content for d in all_docs])
    else:
        print("Creating new index...")
        index.add([d.page_content for d in all_docs])

    print("✅ Index ready at:", index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True, help="Files/folders/URLs to ingest")
    args = parser.parse_args()
    main(args.paths)
