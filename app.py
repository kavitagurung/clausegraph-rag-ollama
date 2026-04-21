from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_PATH = DATA_DIR / "index.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"documents": {}, "chunks": []}

    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_index(store: dict[str, Any]) -> None:
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=True, indent=2)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_file(path)
    return read_text_file(path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            boundary = normalized.rfind(" ", start + int(chunk_size * 0.6), end)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator


def ollama_tags() -> dict[str, Any]:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=15)
    response.raise_for_status()
    return response.json()


def ollama_embed(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": texts},
        timeout=180,
    )

    if response.ok:
        payload = response.json()
        embeddings = payload.get("embeddings")
        if embeddings:
            return embeddings

    fallback_embeddings: list[list[float]] = []
    for text in texts:
        fallback = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=180,
        )
        fallback.raise_for_status()
        fallback_embeddings.append(fallback.json()["embedding"])

    return fallback_embeddings


def ollama_generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[{item['source_label']}] {item['text']}" for item in contexts
    )
    return (
        "You are a document question-answering assistant.\n"
        "Answer only from the provided context.\n"
        "If the answer is not supported by the context, say: "
        "'I do not know from the indexed documents.'\n"
        "End with a short 'Sources:' line using the provided source labels.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def unique_upload_path(filename: str) -> Path:
    candidate = UPLOAD_DIR / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        next_candidate = UPLOAD_DIR / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def document_summary(filename: str, store: dict[str, Any]) -> dict[str, Any]:
    path = UPLOAD_DIR / filename
    metadata = store["documents"].get(filename, {})
    chunk_count = sum(1 for chunk in store["chunks"] if chunk["filename"] == filename)
    return {
        "filename": filename,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "indexed_chunks": chunk_count,
        "updated_at": metadata.get("updated_at"),
    }


def index_document(filename: str, chunk_size: int, overlap: int, store: dict[str, Any]) -> dict[str, Any]:
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} does not exist")

    raw_text = read_document(path)
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    embeddings = ollama_embed(chunks) if chunks else []

    store["chunks"] = [chunk for chunk in store["chunks"] if chunk["filename"] != filename]

    for index, (text, embedding) in enumerate(zip(chunks, embeddings)):
        store["chunks"].append(
            {
                "id": f"{filename}::chunk::{index}",
                "filename": filename,
                "chunk_index": index,
                "text": text,
                "embedding": embedding,
                "source_label": f"{filename}#chunk-{index}",
            }
        )

    store["documents"][filename] = {
        "updated_at": utc_now(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
    }

    save_index(store)
    return document_summary(filename, store)


def retrieve_chunks(question: str, top_k: int, filename: str | None, store: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = store["chunks"]
    if filename:
        candidates = [chunk for chunk in candidates if chunk["filename"] == filename]

    if not candidates:
        return []

    query_embedding = ollama_embed([question])[0]
    ranked = []
    for chunk in candidates:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        ranked.append({**chunk, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


@app.get("/")
def serve_index() -> Any:
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health() -> Any:
    try:
        tags = ollama_tags()
        return jsonify(
            {
                "ok": True,
                "ollama_base_url": OLLAMA_BASE_URL,
                "chat_model": OLLAMA_CHAT_MODEL,
                "embed_model": OLLAMA_EMBED_MODEL,
                "available_models": [item["name"] for item in tags.get("models", [])],
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "ollama_base_url": OLLAMA_BASE_URL,
                "chat_model": OLLAMA_CHAT_MODEL,
                "embed_model": OLLAMA_EMBED_MODEL,
                "error": str(exc),
            }
        ), 503


@app.get("/api/documents")
def list_documents() -> Any:
    store = load_index()
    filenames = sorted(path.name for path in UPLOAD_DIR.iterdir() if path.is_file())
    return jsonify({"documents": [document_summary(name, store) for name in filenames]})


@app.post("/api/upload")
def upload_documents() -> Any:
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files were uploaded."}), 400

    saved: list[dict[str, Any]] = []
    for incoming in files:
        original_name = secure_filename(incoming.filename or "")
        if not original_name:
            continue

        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        target = unique_upload_path(original_name)
        incoming.save(target)
        saved.append({"filename": target.name, "size_bytes": target.stat().st_size})

    if not saved:
        return jsonify({"error": "No valid files were uploaded."}), 400

    return jsonify({"uploaded": saved}), 201


@app.post("/api/index")
def index_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")
    chunk_size = int(payload.get("chunk_size") or DEFAULT_CHUNK_SIZE)
    overlap = int(payload.get("overlap") or DEFAULT_CHUNK_OVERLAP)

    if overlap >= chunk_size:
        return jsonify({"error": "Overlap must be smaller than chunk size."}), 400

    if filename:
        filenames = [filename]
    else:
        filenames = sorted(path.name for path in UPLOAD_DIR.iterdir() if path.is_file())

    if not filenames:
        return jsonify({"error": "No uploaded files found to index."}), 400

    store = load_index()
    indexed = []
    for name in filenames:
        indexed.append(index_document(name, chunk_size, overlap, store))

    return jsonify({"indexed": indexed})


@app.post("/api/query")
def query_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    filename = payload.get("filename") or None
    top_k = int(payload.get("top_k") or 4)

    if not question:
        return jsonify({"error": "Question is required."}), 400

    store = load_index()
    matches = retrieve_chunks(question, top_k=top_k, filename=filename, store=store)
    if not matches:
        return jsonify({"answer": "I do not know from the indexed documents.", "sources": []})

    answer = ollama_generate(build_prompt(question, matches))
    sources = [
        {
            "id": item["id"],
            "filename": item["filename"],
            "chunk_index": item["chunk_index"],
            "score": round(item["score"], 4),
            "text": item["text"],
            "source_label": item["source_label"],
        }
        for item in matches
    ]
    return jsonify({"answer": answer, "sources": sources})


@app.delete("/api/documents/<path:filename>")
def delete_document(filename: str) -> Any:
    path = UPLOAD_DIR / filename
    store = load_index()

    if path.exists():
        path.unlink()

    store["chunks"] = [chunk for chunk in store["chunks"] if chunk["filename"] != filename]
    store["documents"].pop(filename, None)
    save_index(store)
    return jsonify({"deleted": filename})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
