

"""
ingest.py -- Build Ruth's ChromaDB knowledge base.

Run:  python ingest.py

You do not need to set any paths. If the folder below is wrong, this
script searches your user folder for a 'texts' folder containing PDFs
and uses that instead. Once it finds it, the location is saved to
ruth_paths.json so ruth.py uses the same one.
"""

import os
import re
import sys
import json
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

try:
    from pypdf import PdfReader
except ImportError:
    print("Missing pypdf.  Run:  python -m pip install pypdf")
    sys.exit(1)


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

# First guess. If it is wrong, the script finds the real one itself.
PROJECT_GUESS = Path(r"C:\Users\succe\Bible scholar")

SEARCH_ROOT = Path(r"C:\Users\succe")     # where to hunt if the guess fails
CONFIG_FILE = Path(__file__).parent / "ruth_paths.json"

COLLECTION  = "ruth_library"
EMBED_MODEL = "all-MiniLM-L6-v2"          # MUST be identical in ruth.py

CHUNK_CHARS = 1200
OVERLAP     = 200
BATCH_SIZE  = 128

REBUILD_ALL = False                       # True = wipe and start over


# ---------------------------------------------------------------------
# FINDING YOUR FOLDER
# ---------------------------------------------------------------------

SKIP_DIRS = {
    "AppData", "Application Data", "node_modules", "__pycache__",
    "site-packages", "Windows", "Program Files", "chroma",
}


def has_readable_files(folder):
    try:
        for f in os.listdir(folder):
            if f.lower().endswith((".pdf", ".txt")):
                return True
    except OSError:
        pass
    return False


def hunt_for_texts():
    """Walk the user folder looking for a 'texts' folder with PDFs in it."""
    hits = []
    if not SEARCH_ROOT.exists():
        return hits

    print(f"Searching {SEARCH_ROOT} for your texts folder...")
    for dirpath, dirnames, filenames in os.walk(SEARCH_ROOT):
        here = Path(dirpath)
        try:
            depth = len(here.relative_to(SEARCH_ROOT).parts)
        except ValueError:
            continue
        if depth > 4:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        if here.name.lower() == "texts":
            if any(f.lower().endswith((".pdf", ".txt")) for f in filenames):
                hits.append(here)
    return hits


def resolve_paths():
    """Return (texts_dir, chroma_dir) or (None, None)."""
    guess = PROJECT_GUESS / "texts"
    if guess.exists() and has_readable_files(guess):
        print(f"Found your texts folder: {guess}")
        return guess, PROJECT_GUESS / "chroma"

    print(f"Not at the expected path: {guess}")
    hits = hunt_for_texts()

    if not hits:
        print("\nCould not find any folder named 'texts' containing PDFs.")
        print("Folders directly inside your user folder:")
        try:
            for name in sorted(os.listdir(SEARCH_ROOT)):
                if (SEARCH_ROOT / name).is_dir() and not name.startswith("."):
                    print("   ", name)
        except OSError as e:
            print("   (could not list:", e, ")")
        return None, None

    if len(hits) > 1:
        print("\nFound more than one. Using the first:")
        for h in hits:
            print("   ", h)

    texts_dir = hits[0]
    print(f"\nUsing: {texts_dir}")
    return texts_dir, texts_dir.parent / "chroma"


# ---------------------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------------------

def clean(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path):
    pages = []
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        print("   !! could not open:", e)
        return pages

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        txt = clean(raw)
        if len(txt) > 40:
            pages.append((i, txt))
    return pages


def read_txt(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return [(1, clean(path.read_text(encoding=enc)))]
        except UnicodeDecodeError:
            continue
    print("   !! could not decode:", path.name)
    return []


def chunk_pages(pages):
    buf = ""
    start_page = None
    end_page = None

    for page_no, text in pages:
        if start_page is None:
            start_page = page_no
        end_page = page_no
        buf = (buf + "\n\n" + text).strip() if buf else text

        while len(buf) >= CHUNK_CHARS:
            cut = buf.rfind(" ", 0, CHUNK_CHARS)
            if cut < CHUNK_CHARS // 2:
                cut = CHUNK_CHARS
            piece = buf[:cut].strip()
            if piece:
                yield piece, start_page, end_page
            buf = buf[max(0, cut - OVERLAP):].strip()
            start_page = page_no

    if len(buf) > 100:
        yield buf, start_page or 1, end_page or 1


def make_id(source, index):
    h = hashlib.md5(f"{source}::{index}".encode("utf-8")).hexdigest()[:16]
    return f"{h}-{index}"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    texts_dir, chroma_dir = resolve_paths()
    if texts_dir is None:
        print("\nStopping. Move your PDFs into a folder named 'texts',")
        print("or edit PROJECT_GUESS at the top of this file.")
        return

    chroma_dir.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps({"texts": str(texts_dir), "chroma": str(chroma_dir)}, indent=2),
        encoding="utf-8",
    )
    print("Saved these locations to", CONFIG_FILE.name)

    print("\nLoading embedding model:", EMBED_MODEL)
    print("(first run downloads about 90 MB, quietly -- be patient)")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=str(chroma_dir))

    if REBUILD_ALL:
        try:
            client.delete_collection(COLLECTION)
            print("Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    files = sorted(
        p for p in texts_dir.rglob("*")
        if p.suffix.lower() in (".pdf", ".txt")
    )
    print(f"\nFound {len(files)} files to read.\n")

    total_new = 0
    skipped_empty = 0

    for path in files:
        source = path.name

        try:
            if collection.get(where={"source": source}, limit=1)["ids"]:
                print(f"-- {source}  (already ingested)")
                continue
        except Exception:
            pass

        print(f">> {source}")
        pages = read_pdf(path) if path.suffix.lower() == ".pdf" else read_txt(path)

        if not pages:
            print("   no readable text (scanned image PDF?), skipped")
            skipped_empty += 1
            continue

        docs, metas, ids = [], [], []
        for i, (text, p_start, p_end) in enumerate(chunk_pages(pages)):
            docs.append(text)
            metas.append({
                "source": source,
                "title": path.stem,
                "page_start": p_start,
                "page_end": p_end,
            })
            ids.append(make_id(source, i))

        for i in range(0, len(docs), BATCH_SIZE):
            collection.add(
                documents=docs[i:i + BATCH_SIZE],
                metadatas=metas[i:i + BATCH_SIZE],
                ids=ids[i:i + BATCH_SIZE],
            )
            print(f"   embedded {min(i + BATCH_SIZE, len(docs))}/{len(docs)} chunks")

        total_new += len(docs)

    print(f"\nDone. Added {total_new} new chunks.")
    if skipped_empty:
        print(f"{skipped_empty} files had no extractable text (they need OCR).")
    print("Collection now holds", collection.count(), "chunks.")


if __name__ == "__main__":
    main()

 
