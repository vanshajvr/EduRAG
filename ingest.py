import argparse
if path.startswith("http://") or path.startswith("https://"):
loader = UnstructuredURLLoader(urls=[path])
docs.extend(loader.load())
elif os.path.isdir(path):
for root, _, files in os.walk(path):
for fn in files:
ext = os.path.splitext(fn)[1].lower()
full = os.path.join(root, fn)
if ext in SUPPORTED_EXT:
docs.extend(TextLoader(full, encoding='utf-8').load())
elif ext == ".pdf":
docs.extend(PyPDFLoader(full).load())
else:
ext = os.path.splitext(path)[1].lower()
if ext in SUPPORTED_EXT:
docs.extend(TextLoader(path, encoding='utf-8').load())
elif ext == ".pdf":
docs.extend(PyPDFLoader(path).load())
else:
raise ValueError(f"Unsupported file: {path}")
# Attach source metadata
for d in docs:
if 'source' not in d.metadata:
d.metadata['source'] = path
return docs




def main(paths: List[str]):
all_docs = []
for p in paths:
print(f"Loading {p}...")
all_docs.extend(load_path(p))
if not all_docs:
print("No documents found.")
return
index = VectorIndex()
if os.path.isdir(os.getenv("INDEX_DIR", "data/index")) and os.path.isfile(os.path.join(os.getenv("INDEX_DIR", "data/index"), "index.faiss")):
print("Adding to existing index...")
index.add(all_docs)
else:
print("Creating new index...")
index.create(all_docs)
print("Index ready.")




if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--paths", nargs='+', required=True, help="Files/folders/URLs to ingest")
args = parser.parse_args()
main(args.paths)
