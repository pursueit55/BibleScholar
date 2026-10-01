

import pickle

CHUNKS_PATH = "ruth_scripture_chunks.pkl"

with open(CHUNKS_PATH, 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
sources = data['sources']

unique_sources = sorted(set(sources))
print(f"Total chunks: {len(chunks)}")
print(f"Total unique files: {len(unique_sources)}\n")

print("Files included:")
for s in unique_sources:
    count = sources.count(s)
    print(f"  {s} — {count} chunks") 
