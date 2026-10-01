

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import os
import pickle

TEXTS_FOLDER = "texts"
INDEX_SAVE_PATH = "ruth_scripture_index.faiss"
CHUNKS_SAVE_PATH = "ruth_scripture_chunks.pkl"

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def load_and_chunk(folder_path, chunk_size=400, overlap=50):
    chunks = []
    sources = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif filename.endswith('.pdf'):
            print(f"Extracting text from {filename}...")
            text = extract_pdf_text(filepath)
        else:
            continue  # skip anything that's not .txt or .pdf

        words = text.split()
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            sources.append(filename)
            i += chunk_size - overlap

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
