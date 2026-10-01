

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

INDEX_PATH = "ruth_scripture_index.faiss"
CHUNKS_PATH = "ruth_scripture_chunks.pkl"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    sources = data['sources']

def retrieve_context(query, top_k=4):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    results = []
    for i in indices[0]:
        results.append({"text": chunks[i], "source": sources[i]})
    return results

def format_context_for_prompt(results):
    formatted = ""
    for r in results:
        formatted += f"\n[From {r['source']}]\n{r['text']}\n"
    return formatted 
