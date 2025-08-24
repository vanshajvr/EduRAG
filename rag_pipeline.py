import os
if self.vs is None:
return self.create(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = text_splitter.split_documents(docs)
self._vs.add_documents(chunks)
self._vs.save_local(self.index_dir)


def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
if self.vs is None:
raise RuntimeError("Index not built. Run ingest.py first.")
docs = self._vs.similarity_search_with_score(query, k=k)
out = []
for d, score in docs:
out.append(RetrievedChunk(content=d.page_content, source=d.metadata.get("source", ""), score=float(score), metadata=d.metadata))
return out


class Generator:
def __init__(self, model_name: str = MODEL_NAME):
self.client = OpenAI()
self.model = model_name


def generate(self, prompt: str, system: str = "You are a helpful, precise study assistant.") -> str:
resp = self.client.chat.completions.create(
model=self.model,
messages=[
{"role": "system", "content": system},
{"role": "user", "content": prompt},
],
temperature=0.2,
)
return resp.choices[0].message.content


class RAGPipeline:
def __init__(self, index: VectorIndex, gen: Optional[Generator] = None):
self.index = index
self.gen = gen


def answer(self, question: str, k: int = 5) -> Dict[str, Any]:
retrieved = self.index.retrieve(question, k=k)
context_blocks = []
for r in retrieved:
snippet = r.content[:500]
context_blocks.append(f"Source: {r.source}\nSnippet: {snippet}")
context_str = "\n\n".join(context_blocks)


prompt = f"""
You'll need to answer strictly from the provided sources. If the answer is not in the sources, say you don't know.


Question: {question}


Sources:
{context_str}


Could you answer with citations as [S1], [S2]... mapping to the numbered sources below, and then list the sources you used?
"""
if self.gen is None:
return {
"answer": "LLM not configured. Showing retrieved contexts only.",
"contexts": [r.__dict__ for r in retrieved],
}
raw = self.gen.generate(prompt)
result = {
"answer": raw,
"contexts": [r.__dict__ for r in retrieved],
}
return result
