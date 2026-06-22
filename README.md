# Vector RAG Demo

A small local Retrieval-Augmented Generation project that indexes documents from
`data/` into Chroma, retrieves relevant chunks with sentence-transformer
embeddings, and answers with a selectable LLM provider: local Ollama or
DeepSeek API.

## Quick Start

```powershell
venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3:latest
python rag.py ingest --reset
python rag.py ask "What is this document about?"
streamlit run app.py
```

## How It Works

1. Put `.pdf`, `.txt`, or `.md` files in `data/`.
2. Run ingestion to chunk and embed the files.
3. Ask a question from the CLI or Streamlit app.
4. The selected LLM provider generates a cited answer from the retrieved context.

Optional environment variables:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="llama3:latest"

$env:LLM_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="[ENCRYPTION_KEY]"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

Or create a local `.env` file from `.env.example`. The app loads `.env`
automatically, and `.env` is ignored by Git.

You can also choose the provider in the Streamlit sidebar or from the CLI:

```powershell
python rag.py ask "What is this document about?" --provider deepseek --model deepseek-v4-flash
```
