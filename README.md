# clausegraph-rag-ollama

ClauseGraph RAG is now a clean Ollama-based retrieval app with:

- A Python backend built with Flask
- A browser UI in `index.html`
- Local document upload and delete flows
- Semantic retrieval using Ollama embeddings
- Answer generation with source chunks

## Why the old repo was broken

The previous repo mixed a Streamlit server app with GitHub Pages workflows. GitHub Pages can only host static files, so it could never run the actual RAG backend. That is why the deployed site showed static content instead of a working app.

## New project structure

```text
app.py            Flask backend and RAG API
index.html        HTML UI with embedded CSS and JavaScript
requirements.txt  Python dependencies
data/uploads      Uploaded source files
data/index.json   Local persisted chunk + embedding index
```

## Features

- Upload `.pdf`, `.txt`, and `.md` files
- Chunk and embed documents with Ollama
- Query all indexed files or a single selected file
- Inspect retrieved source chunks in the UI
- Delete files and their indexed vectors

## Requirements

- Python 3.10+
- Ollama running locally or remotely
- An embedding model such as `nomic-embed-text`
- A chat model such as `llama3.2:3b`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
python3 app.py
```

Open `http://localhost:8000`.

## Environment variables

- `OLLAMA_BASE_URL` default: `http://localhost:11434`
- `OLLAMA_CHAT_MODEL` default: `llama3.2:3b`
- `OLLAMA_EMBED_MODEL` default: `nomic-embed-text`
- `RAG_CHUNK_SIZE` default: `900`
- `RAG_CHUNK_OVERLAP` default: `180`
- `PORT` default: `8000`

## API endpoints

- `GET /api/health`
- `GET /api/documents`
- `POST /api/upload`
- `POST /api/index`
- `POST /api/query`
- `DELETE /api/documents/<filename>`

## Notes

- This app is meant to run as a real backend service, not on GitHub Pages.
- The local JSON index keeps the setup simple and easy to understand.
- If you want, the next step can be upgrading this to ChromaDB, SQLite, or a production deployment target.
