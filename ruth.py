



"""
ruth.py -- Ruth, the biblical scholar.

Run:  python ruth.py

SPEED VERSION
- Context locked to 8192 so the model stays small in memory.
- Only 3 passages per question, each trimmed to a sensible length.
- The model stays loaded between questions (no reload wait).
- The model and the library are warmed up at startup, so your first
  question does not pay the loading cost.
- Ruth streams: she starts printing and speaking after her first
  sentence instead of waiting for the whole answer.
- Small talk ("thanks", "hello") skips the library entirely.
  Follow-ups ("tell me more", "go on") reuse the last passages.

Commands:  books          list what is in her library
           quiet / loud   voice off and on
           quit
"""

import re
import sys
import json
import time
import queue
import threading
import requests
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

CONFIG_FILE   = Path(__file__).parent / "ruth_paths.json"
PROJECT_GUESS = Path(r"C:\Users\succe\biblescholar")   # fallback only

COLLECTION  = "ruth_library"
EMBED_MODEL = "all-MiniLM-L6-v2"     # MUST match ingest.py

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
TEMPERATURE  = 0.7
MAX_TOKENS   = 220

# --- speed settings ---
NUM_CTX    = 8192      # context window. 32768 made the model 6.1 GB.
KEEP_ALIVE = "30m"     # keep the model loaded this long between questions
USE_GPU    = False     # GT 720 is too small to help. Set True once the
                       # GTX 1660 is powered and the driver is updated
                       # (and remove the CUDA_VISIBLE_DEVICES variable).

FETCH_K           = 20     # candidates pulled before filtering
TOP_K             = 3      # passages actually given to her
MAX_PER_SOURCE    = 2      # most chunks any single book may contribute
MAX_PASSAGE_CHARS = 1200   # each passage is trimmed to about this length
MAX_DISTANCE      = 0.80   # meaning matches weaker than this are left out
                           # (0 = perfect match, 1 = unrelated)
KEYWORD_NAMES        = 3     # how many names from a question get a keyword search
KEYWORD_MAX_DISTANCE = 0.95  # name matches can be looser, since the name itself is there
HISTORY_TURNS     = 4      # past exchanges she remembers in the chat

SHOW_SOURCES = True
SHOW_TIMING  = True

SPEAK = True
VOICE = "bf_emma"
# British female: bf_emma  bf_isabella  bf_alice  bf_lily
# British male:   bm_george  bm_daniel  bm_lewis  bm_fable
# American female: af_heart  af_bella  af_nicole  af_sarah  af_sky
# American male:   am_michael  am_adam  am_eric  am_liam

USE_MIC      = False
WHISPER_SIZE = "base"

# Short messages that never need the library.
SMALL_TALK = {
    "thanks", "thank you", "thank you ruth", "thanks ruth", "ok", "okay",
    "hello", "hi", "hey", "hello ruth", "hi ruth", "good morning",
    "good evening", "good night", "amen", "wow", "interesting", "i see",
}

# Follow-ups that should reuse the passages from the last question.
FOLLOW_UPS = {
    "tell me more", "go on", "say more", "keep going", "continue",
    "what else", "why", "why is that", "how so", "explain that",
    "and then", "what happened next", "really",
}

BAIT = ("would you like", "shall i", "let me know if",
        "i could also", "do you want me to", "if you'd like")


# ---------------------------------------------------------------------
# HER CHARACTER
# ---------------------------------------------------------------------

IDENTITY = """You are Ruth, a scholar of the Hebrew scriptures, the Second Temple period, the Dead Sea Scrolls, and the writings of Josephus. You have spent your life in these texts.

HOW YOU SPEAK
- You are having a conversation, not delivering a lecture. Two to five sentences is a normal answer.
- Answer only what was asked. Nothing more.
- Never end your turn with an offer. No "would you like to know more", no "I could also tell you about", no suggesting other topics, no follow-up questions unless something was genuinely unclear.
- Never mention documents, sources, passages provided to you, context, retrieval, or "the text says". What you know, you simply know.
- No bullet points, no headings, no numbered lists, no bold. Plain spoken sentences only.
- You are warm and direct, not reverent or ceremonial.

HONESTY
- Never say you have no information, no source, or no record of something. You are a scholar; answer from what you know.
- If a detail is uncertain or you are unsure of it, say so in plain words, for example "I'm not certain of the exact year."
- Where scholars genuinely disagree, say that they disagree and give the shape of the argument. Do not present one camp's reading as settled.
- When a writer speaks of himself in the first person ("I, Josephus...") or in the third person by name, that writer is the person being asked about. Speak about him from his own words."""

RECALL_HEADER = """WHAT YOU ARE RECALLING RIGHT NOW
The following are your own memories from years of study. Build your answer from them. Do not quote them at length and never refer to them as anything other than what you know. If they only partly answer the question, use what they do say and be plain about what you are less sure of."""

EXAMPLES = [
    {"role": "user", "content": "Who was Josephus?"},
    {"role": "assistant", "content": "A Jewish priest and commander who surrendered to the Romans during the revolt and spent the rest of his life writing under their patronage. That double position is why people still argue about him -- he was an eyewitness, but an eyewitness with a patron to please."},
    {"role": "user", "content": "Was the War Scroll meant literally?"},
    {"role": "assistant", "content": "Hard to say with certainty. It reads like a battle manual, with formations and trumpet signals laid out in detail, but the enemy is cosmic and the outcome is already decided. My own sense is that the community expected a real war and understood it as the visible edge of a spiritual one."},
]


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

def find_chroma():
    if CONFIG_FILE.exists():
        try:
            p = Path(json.loads(CONFIG_FILE.read_text(encoding="utf-8"))["chroma"])
            if p.exists():
                return p
        except Exception:
            pass

    fallback = PROJECT_GUESS / "chroma"
    if fallback.exists():
        return fallback

    print("Cannot find Ruth's library folder. Run ingest.py first.")
    sys.exit(1)


# ---------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------

def open_collection(chroma_dir):
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


def trim_passage(text):
    """Cut a passage down to MAX_PASSAGE_CHARS, ending on a full sentence."""
    text = " ".join(text.split())
    if len(text) <= MAX_PASSAGE_CHARS:
        return text
    cut = text[:MAX_PASSAGE_CHARS]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end > MAX_PASSAGE_CHARS * 0.5:
        return cut[:end + 1]
    return cut + "..."


STOP_WORDS = {
    "about", "tell", "what", "when", "where", "which", "while", "would",
    "could", "should", "there", "their", "these", "those", "does", "did",
    "have", "with", "from", "that", "this", "your", "know", "explain",
    "describe", "please", "ruth", "why", "how", "who", "whom", "whose",
    "the", "and", "was", "were", "are", "is", "his", "her", "him", "they",
    "them", "said", "say", "write", "wrote", "written", "book", "books",
}


def key_names(question):
    """
    Pick out the words most likely to be names: Josephus, Herod, Qumran.
    These get a keyword search so passages that literally mention the
    name are found, even when the meaning search misses them.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    picks = []
    for w in words:
        base = w.strip("'-")
        if len(base) < 4 or base.lower() in STOP_WORDS:
            continue
        name = base[0].upper() + base[1:].lower()
        if name not in picks:
            picks.append(name)
    picks.sort(key=len, reverse=True)
    return picks[:KEYWORD_NAMES]


def search(collection, question, name=None):
    """One search. With a name, only passages containing that word."""
    kwargs = {"query_texts": [question], "n_results": FETCH_K}
    if name:
        kwargs["where_document"] = {"$contains": name}
    try:
        res = collection.query(**kwargs)
    except Exception:
        return []
    return list(zip(res.get("documents", [[]])[0],
                    res.get("metadatas", [[]])[0],
                    res.get("distances", [[]])[0]))


def recall(collection, question):
    """
    Two searches, merged:
      1. NAME search -- passages that actually contain the names in the
         question (Josephus, Herod...). Finds the man in his own words.
      2. MEANING search -- passages about the same idea, even if they use
         different words.
    Name hits go first. At most MAX_PER_SOURCE chunks from any one book,
    TOP_K chunks in total.
    """
    candidates = []
    seen = set()

    for name in key_names(question):
        for doc, meta, dist in search(collection, question, name):
            if doc in seen or dist > KEYWORD_MAX_DISTANCE:
                continue
            seen.add(doc)
            candidates.append(("name", doc, meta, dist))

    for doc, meta, dist in search(collection, question):
        if doc in seen:
            continue
        seen.add(doc)
        candidates.append(("mean", doc, meta, dist))

    name_hits = sorted([c for c in candidates if c[0] == "name"], key=lambda c: c[3])
    mean_hits = sorted([c for c in candidates if c[0] == "mean"], key=lambda c: c[3])

    chosen, debug, per_source = [], [], {}

    for kind, doc, meta, dist in name_hits + mean_hits:
        if len(chosen) >= TOP_K:
            break
        meta = meta or {}
        title = meta.get("title", "unknown")
        pages = f"p{meta.get('page_start')}-{meta.get('page_end')}"
        tag = "NAME" if kind == "name" else "USED"

        if kind == "mean" and dist > MAX_DISTANCE:
            debug.append(f"   [weak] {dist:.3f}  {title} {pages}")
            continue
        if per_source.get(title, 0) >= MAX_PER_SOURCE:
            debug.append(f"   [cap ] {dist:.3f}  {title} {pages}")
            continue

        per_source[title] = per_source.get(title, 0) + 1
        chosen.append(trim_passage(doc))
        debug.append(f"   [{tag}] {dist:.3f}  {title} {pages}")

    names = key_names(question)
    if names:
        debug.insert(0, f"   name search: {', '.join(names)}")
    if len(per_source) == 1 and chosen:
        debug.append("   (only one book matched -- her answer rests on it alone)")

    return chosen, debug


def classify(question):
    """Decide whether a message needs the library."""
    norm = re.sub(r"[^\w\s']", "", question.lower()).strip()
    if norm in SMALL_TALK:
        return "chat"
    if norm in FOLLOW_UPS:
        return "follow"
    return "search"


def build_system(passages):
    if not passages:
        return IDENTITY
    return f"{IDENTITY}\n\n{RECALL_HEADER}\n\n" + "\n\n---\n\n".join(passages)


def list_books(collection):
    """Print every distinct source and its chunk count."""
    print("  reading the shelf list, this takes a moment...")
    try:
        got = collection.get(include=["metadatas"])
    except Exception as e:
        print("   could not read:", e)
        return
    counts = {}
    for m in got.get("metadatas", []):
        t = (m or {}).get("title", "unknown")
        counts[t] = counts.get(t, 0) + 1
    for title in sorted(counts, key=lambda k: -counts[k]):
        print(f"   {counts[title]:>6}  {title}")
    print(f"   {len(counts)} books, {sum(counts.values())} chunks\n")


# ---------------------------------------------------------------------
# THE BRAIN
# ---------------------------------------------------------------------

def model_options(num_predict=MAX_TOKENS):
    opts = {
        "temperature": TEMPERATURE,
        "num_predict": num_predict,
        "repeat_penalty": 1.1,
        "num_ctx": NUM_CTX,
    }
    if not USE_GPU:
        opts["num_gpu"] = 0
    return opts


def warm_up(collection):
    """Load the model and the embedder now, so question one is fast."""
    print("Warming up (loading the model into memory)...")
    t0 = time.time()
    try:
        collection.query(query_texts=["warm up"], n_results=1)
    except Exception:
        pass
    try:
        requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": model_options(num_predict=1),
        }, timeout=600)
        print(f"Ready in {time.time() - t0:.1f}s.\n")
    except requests.exceptions.ConnectionError:
        print("(Ollama is not running. Start the Ollama app, then run Ruth again.)\n")
    except Exception as e:
        print(f"(Warm-up problem: {e})\n")


def stream_ollama(system_text, history, question):
    """Yield Ruth's reply a few words at a time as the model writes it."""
    messages = [{"role": "system", "content": system_text}]
    messages.extend(EXAMPLES)
    messages.extend(history[-HISTORY_TURNS * 2:])
    messages.append({"role": "user", "content": question})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": model_options(),
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")
MIN_SENTENCE_CHARS = 25


def sentences(pieces):
    """Group streamed words into whole sentences."""
    buf = ""
    for piece in pieces:
        buf += piece
        while True:
            m = SENTENCE_END.search(buf, MIN_SENTENCE_CHARS)
            if not m:
                break
            yield buf[:m.end()].strip()
            buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()


def clean(sentence):
    s = re.sub(r"[*_#`]", "", sentence)
    s = re.sub(r"^\s*[-\u2022]\s*", "", s)
    return " ".join(s.split())


def is_bait(sentence):
    low = sentence.lower()
    return low.endswith("?") and any(b in low for b in BAIT)


# ---------------------------------------------------------------------
# HER VOICE
# ---------------------------------------------------------------------

class Voice:
    """Speaks sentences in the background while the model keeps writing."""

    def __init__(self):
        self.ok = False
        self.pipeline = None
        self.q = None
        if not SPEAK:
            return
        try:
            from kokoro import KPipeline
            import sounddevice as sd
            self.sd = sd
            lang = "b" if VOICE.startswith(("bf_", "bm_")) else "a"
            self.pipeline = KPipeline(lang_code=lang)
            self.q = queue.Queue()
            threading.Thread(target=self._worker, daemon=True).start()
            self.ok = True
            print(f"Voice ready: {VOICE}")
        except Exception as e:
            print(f"(Voice disabled: {e})")

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                if self.ok:
                    for _, _, audio in self.pipeline(text, voice=VOICE, speed=1.0):
                        if not self.ok:
                            break
                        self.sd.play(audio, 24000)
                        self.sd.wait()
            except Exception as e:
                print(f"\n(Speech error: {e})")
            finally:
                self.q.task_done()

    def say(self, text):
        if self.ok and text and self.q is not None:
            self.q.put(text)

    def finish(self):
        """Wait until she has finished speaking."""
        if self.q is not None:
            self.q.join()


# ---------------------------------------------------------------------
# HER EARS
# ---------------------------------------------------------------------

class Ears:
    def __init__(self):
        self.ok = False
        if not USE_MIC:
            return
        try:
            import whisper
            import sounddevice as sd
            import numpy as np
            self.sd, self.np = sd, np
            print(f"Loading Whisper ({WHISPER_SIZE})...")
            self.model = whisper.load_model(WHISPER_SIZE, device="cpu")
            self.ok = True
        except Exception as e:
            print(f"(Microphone disabled: {e})")

    def listen(self):
        q = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata.copy())

        print("Recording -- press Enter to stop.")
        with self.sd.InputStream(samplerate=16000, channels=1,
                                 dtype="float32", callback=cb):
            input()

        frames = []
        while not q.empty():
            frames.append(q.get())
        if not frames:
            return ""

        audio = self.np.concatenate(frames, axis=0).flatten()
        return self.model.transcribe(audio, fp16=False).get("text", "").strip()


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

def main():
    global SPEAK

    chroma_dir = find_chroma()
    print("Opening Ruth's library at", chroma_dir)
    collection = open_collection(chroma_dir)
    print(f"{collection.count()} chunks in memory.")

    voice = Voice()
    ears = Ears()
    warm_up(collection)

    history = []
    last_passages = []

    print("Ruth is here. Type 'books' to see her shelf, 'quit' to leave.\n")

    while True:
        if ears.ok:
            input("Press Enter to speak... ")
            question = ears.listen()
            if not question:
                continue
            print(f"You: {question}")
        else:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if not question:
            continue

        low = question.lower()
        if low in ("quit", "exit", "goodbye"):
            break
        if low == "books":
            list_books(collection)
            continue
        if low == "quiet":
            SPEAK = False
            voice.ok = False
            print("(voice off)\n")
            continue
        if low == "loud":
            SPEAK = True
            if voice.pipeline is not None:
                voice.ok = True
            else:
                voice = Voice()
            print("(voice on)\n")
            continue

        kind = classify(question)
        if kind == "chat":
            passages = []
        elif kind == "follow":
            passages = last_passages
            if SHOW_SOURCES:
                print("  -- recall --\n   (follow-up: using the same passages as before)")
        else:
            passages, debug = recall(collection, question)
            last_passages = passages
            if SHOW_SOURCES:
                print("  -- recall --")
                for line in debug:
                    print(line)
                if not passages:
                    print("   no strong match -- she is on her own here")

        start = time.time()
        first_words = None
        spoken = []

        print("\nRuth: ", end="", flush=True)
        try:
            for sent in sentences(stream_ollama(build_system(passages), history, question)):
                s = clean(sent)
                if not s or is_bait(s):
                    continue
                if first_words is None:
                    first_words = time.time() - start
                print(s + " ", end="", flush=True)
                voice.say(s)
                spoken.append(s)
        except requests.exceptions.ConnectionError:
            print("(Ollama is not running. Start the Ollama app or type: ollama serve)", end="")
        except Exception as e:
            print(f"(Ollama error: {e})", end="")
        print("\n")

        if SHOW_TIMING and first_words is not None:
            print(f"  (first words after {first_words:.1f}s, "
                  f"finished writing in {time.time() - start:.1f}s)\n")

        voice.finish()

        answer = " ".join(spoken)
        if answer:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

    print("\nRuth: Come back when you have another question.")


if __name__ == "__main__":
    main()

 
