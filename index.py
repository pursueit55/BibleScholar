

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

TEXTS_FOLDER = "path/to/your/text_files"  # folder with War Scroll, Jasher, Josephus, History of the Jews
INDEX_SAVE_PATH = "ruth_scripture_index.faiss"
CHUNKS_SAVE_PATH = "ruth_scripture_chunks.pkl"

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_chunk(folder_path, chunk_size=400, overlap=50):
    chunks = []
    sources = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            words = text.split()
            i = 0
            while i < len(words):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
                sources.append(filename)
                i += chunk_size - overlap  # overlap keeps context from getting cut awkwardly
    return chunks, sources

def build_and_save_index():
    print("Loading and chunking texts...")
    chunks, sources = load_and_chunk(TEXTS_FOLDER)
    print(f"Created {len(chunks)} chunks from {len(set(sources))} files.")

    print("Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(CHUNKS_SAVE_PATH, 'wb') as f:
        pickle.dump({'chunks': chunks, 'sources': sources}, f)

    print(f"Index saved to {INDEX_SAVE_PATH}")
    print(f"Chunks saved to {CHUNKS_SAVE_PATH}")

if __name__ == "__main__":
    build_and_save_index() 
Displaying "thewarscroll.pdf."