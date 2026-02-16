import streamlit as st
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
#        SETTINGS
# =========================
MODEL = "phi3:mini"

# Use /api/generate (works on more Ollama installs than /api/chat)
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ClauseGraph RAG", layout="wide")


# =========================
#        HELPERS
# =========================
def read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


@st.cache_resource
def get_embedder():
    # Small + fast embedding model
    return SentenceTransformer("intfloat/e5-small-v2")


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection("clausegraph")


def list_uploaded_files() -> List[str]:
    return sorted([p.name for p in UPLOAD_DIR.glob("*")], reverse=True)


def delete_file_and_vectors(filename: str) -> None:
    # delete vectors
    collection = get_chroma_collection()
    collection.delete(where={"source": filename})

    # delete file
    fp = UPLOAD_DIR / filename
    if fp.exists():
        fp.unlink()


def clear_vectors_for_file(filename: str) -> None:
    collection = get_chroma_collection()
    collection.delete(where={"source": filename})


def index_file(filename: str, chunk_size: int, overlap: int) -> int:
    path = UPLOAD_DIR / filename
    if not path.exists():
        return 0

    if path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        text = read_txt(path)

    if not text.strip():
        return 0

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    embedder = get_embedder()
    collection = get_chroma_collection()

    # Remove previous vectors for this file (clean re-index)
    collection.delete(where={"source": filename})

    # Stable IDs (no timestamp) so re-index is predictable
    ids = [f"{filename}::chunk::{i}" for i in range(len(chunks))]

    # E5 prefers passage/query prefixes
    embeddings = embedder.encode(
        [f"passage: {c}" for c in chunks],
        normalize_embeddings=True
    ).tolist()

    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def chroma_vector_retrieve(query: str, top_k: int, scope_filename: Optional[str]) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    collection = get_chroma_collection()

    try:
        if collection.count() == 0:
            return []
    except Exception:
        pass

    q_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]
    where = {"source": scope_filename} if scope_filename else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    out = []
    for i in range(len(results["documents"][0])):
        out.append({
            "id": results["ids"][0][i],
            "doc": results["documents"][0][i],
            "meta": results["metadatas"][0][i] if results.get("metadatas") else {},
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "source": "vector",
        })
    return out


@st.cache_data(show_spinner=False)
def build_tfidf_index(docs: List[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs) if docs else None
    return vectorizer, X


def keyword_retrieve(query: str, top_k: int, scope_filename: Optional[str]) -> List[Dict[str, Any]]:
    collection = get_chroma_collection()
    where = {"source": scope_filename} if scope_filename else None

    got = collection.get(where=where, include=["documents", "metadatas"])

    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    ids = got.get("ids") or []

    if not docs:
        return []

    vectorizer, X = build_tfidf_index(docs)
    if X is None:
        return []

    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()

    top_idx = sims.argsort()[::-1][:top_k]

    out = []
    for i in top_idx:
        if sims[i] <= 0:
            continue
        out.append({
            "id": ids[i],
            "doc": docs[i],
            "meta": metas[i] if i < len(metas) else {},
            "kw_score": float(sims[i]),
            "source": "keyword",
        })
    return out


def hybrid_retrieve(query: str, top_k: int, scope_filename: Optional[str]) -> List[Dict[str, Any]]:
    vec_hits = chroma_vector_retrieve(query, top_k=top_k, scope_filename=scope_filename)
    kw_hits = keyword_retrieve(query, top_k=top_k, scope_filename=scope_filename)

    merged: Dict[str, Dict[str, Any]] = {}

    for h in vec_hits:
        merged[h["id"]] = h

    for h in kw_hits:
        if h["id"] in merged:
            merged[h["id"]]["kw_score"] = h.get("kw_score")
            merged[h["id"]]["source"] = "hybrid"
        else:
            merged[h["id"]] = h

    def rank_key(item: Dict[str, Any]):
        src = item.get("source", "")
        hybrid_bonus = 1 if src == "hybrid" else 0
        kw = item.get("kw_score", 0.0) or 0.0
        dist = item.get("distance", 999.0) if item.get("distance") is not None else 999.0
        return (hybrid_bonus, kw, -dist)

    ranked = sorted(merged.values(), key=rank_key, reverse=True)
    return ranked[:top_k]


def answer_with_context(question: str, context_blocks: List[str], stream: bool = False) -> str:
    system = (
        "You are a helpful assistant. Use ONLY the provided CONTEXT to answer.\n"
        "If the answer is not in the context, say: 'I don't know from the provided documents.'\n"
        "Always include a short Sources section listing the chunk IDs you used."
    )

    context_text = "\n\n".join(context_blocks)

    prompt = (
        f"SYSTEM:\n{system}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }

    if not stream:
        resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    # Streaming: JSON-lines with "response"
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=180)
    resp.raise_for_status()

    full = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            obj = __import__("json").loads(line.decode("utf-8"))
            full += obj.get("response", "")
        except Exception:
            continue
    return full.strip()


# =========================
#   SIDEBAR: CONTROLS
# =========================
st.sidebar.header("📄 Documents")
st.sidebar.caption(f"Uploads folder:\n{UPLOAD_DIR}")

uploaded = st.sidebar.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

# Save upload only once (prevents rerun loops)
if "last_saved_name" not in st.session_state:
    st.session_state.last_saved_name = None

if uploaded is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded.name.replace(" ", "_")
    save_name = f"{timestamp}_{safe_name}"
    save_path = UPLOAD_DIR / save_name

    if st.session_state.last_saved_name != save_name:
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.last_saved_name = save_name
        st.sidebar.success(f"Saved: {save_path.name}")
        st.rerun()

st.sidebar.subheader("Saved files")
collection = get_chroma_collection()
try:
    st.sidebar.caption(f"Indexed chunks in DB: {collection.count()}")
except Exception:
    st.sidebar.caption("Indexed chunks in DB: (unknown)")

files = list_uploaded_files()
if not files:
    st.sidebar.caption("No files uploaded yet.")
    selected_file: Optional[str] = None
else:
    selected_file = st.sidebar.selectbox("Choose a file", files)

colA, colB = st.sidebar.columns(2)

if colA.button("🗑️ Delete file", use_container_width=True):
    if not selected_file:
        st.sidebar.warning("Select a file first.")
    else:
        delete_file_and_vectors(selected_file)
        st.session_state.messages = []
        st.sidebar.success("Deleted file + vectors.")
        st.rerun()

if colB.button("🧽 Clear vectors", use_container_width=True):
    if not selected_file:
        st.sidebar.warning("Select a file first.")
    else:
        clear_vectors_for_file(selected_file)
        st.sidebar.success("Cleared vectors for that file.")
        st.rerun()

st.sidebar.divider()

st.sidebar.subheader("Indexing")
chunk_size = st.sidebar.slider("Chunk size", 400, 1600, 900, 100)
overlap = st.sidebar.slider("Overlap", 0, 400, 120, 10)

if st.sidebar.button("⚡ Index selected file", use_container_width=True):
    if not selected_file:
        st.sidebar.warning("Select a file first.")
    else:
        try:
            with st.spinner("Indexing... (first time may take a minute)"):
                n = index_file(selected_file, chunk_size=chunk_size, overlap=overlap)
            if n == 0:
                st.sidebar.error("No text found (empty/unreadable). Try a TXT file first.")
            else:
                st.sidebar.success(f"Indexed {n} chunks.")
                st.rerun()
        except Exception as e:
            st.sidebar.error("Indexing failed:")
            st.sidebar.exception(e)

if st.sidebar.button("⚡ Index ALL files", use_container_width=True):
    if not files:
        st.sidebar.warning("Upload files first.")
    else:
        total = 0
        try:
            with st.spinner("Indexing ALL files..."):
                for fn in files:
                    total += index_file(fn, chunk_size=chunk_size, overlap=overlap)
            st.sidebar.success(f"Indexed total chunks: {total}")
            st.rerun()
        except Exception as e:
            st.sidebar.error("Indexing ALL failed:")
            st.sidebar.exception(e)

st.sidebar.divider()

st.sidebar.subheader("Retrieval")
scope = st.sidebar.radio("Search scope", ["All files", "Selected file only"])
use_hybrid = st.sidebar.toggle("Hybrid retrieval (Vector + Keyword)", value=True)
top_k = st.sidebar.slider("Top-K chunks", 2, 12, 5, 1)
show_chunks = st.sidebar.toggle("Show retrieved chunks", value=True)
stream_answer = st.sidebar.toggle("Stream answer", value=False)

st.sidebar.divider()
if st.sidebar.button("🧹 Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# =========================
#        MAIN UI
# =========================
st.title("🧠 ClauseGraph RAG (Local AI)")
st.caption("Advanced RAG: upload → index → ask questions with sources. Now with delete + hybrid search.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

effective_scope_file = selected_file if (scope == "Selected file only") else None

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            hits = (
                hybrid_retrieve(prompt, top_k=top_k, scope_filename=effective_scope_file)
                if use_hybrid
                else chroma_vector_retrieve(prompt, top_k=top_k, scope_filename=effective_scope_file)
            )

        if not hits:
            st.markdown("I don't know from the provided documents. (Index a file first.)")
            st.session_state.messages.append({"role": "assistant", "content": "I don't know from the provided documents. (Index a file first.)"})
        else:
            context_blocks = []
            used_ids = []
            for h in hits:
                cid = h["id"]
                doc = h["doc"]
                used_ids.append(cid)
                context_blocks.append(f"[{cid}] {doc}")

            with st.spinner("Answering..."):
                try:
                    reply = answer_with_context(prompt, context_blocks, stream=stream_answer)
                except Exception as e:
                    st.error("Ollama call failed. Make sure Ollama is running and the model is pulled.")
                    st.exception(e)
                    reply = ""

            if reply:
                st.markdown(reply)
                st.caption("Sources: " + ", ".join(used_ids[:6]) + ("..." if len(used_ids) > 6 else ""))

                if show_chunks:
                    with st.expander("Sources (retrieved chunks)"):
                        for h in hits:
                            meta = h.get("meta") or {}
                            st.markdown(f"**{h['id']}**  \n*{meta.get('source','')} — chunk {meta.get('chunk_index','')}*")
                            st.caption(
                                f"retrieval: {h.get('source')} | "
                                f"kw_score={h.get('kw_score', '')} | "
                                f"distance={h.get('distance', '')}"
                            )
                            st.write(h["doc"][:500] + ("..." if len(h["doc"]) > 500 else ""))
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": reply})
