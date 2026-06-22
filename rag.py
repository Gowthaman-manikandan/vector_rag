from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.errors import NotFoundError
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_PROVIDERS = ("ollama", "deepseek")

# --- Agentic RAG: relevance threshold (cosine distance, 0=identical, 1=unrelated) ---
# Chunks with distance > threshold are considered NOT relevant to the query.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))

# Content types supported for LLM generation
CONTENT_TYPES = [
    "email",
    "greeting",
    "marketing",
    "social_media",
    "design_description",
    "custom_template",
]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def _embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_collection() -> Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def read_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((index, text))
    return pages


def read_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = " ".join(text.split())
    return [(1, text)] if text else []


def load_documents(data_dir: Path = DATA_DIR) -> Iterable[tuple[Path, int, str]]:
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page, text in read_pdf(path):
                yield path, page, text
        elif suffix in {".txt", ".md"}:
            for page, text in read_text(path):
                yield path, page, text


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def make_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, page, text in load_documents(data_dir):
        relative_path = str(path.relative_to(data_dir))
        for index, chunk_text in enumerate(split_text(text), start=1):
            digest = hashlib.sha256(
                f"{relative_path}:{page}:{index}:{chunk_text}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest,
                    text=chunk_text,
                    metadata={
                        "source": relative_path,
                        "page": page,
                        "chunk": index,
                    },
                )
            )
    return chunks


def ingest(data_dir: Path = DATA_DIR, reset: bool = False) -> int:
    if reset:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass

    collection = get_collection()
    chunks = make_chunks(data_dir)
    if not chunks:
        return 0

    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)


def retrieve(question: str, n_results: int = 4) -> list[dict[str, object]]:
    collection = get_collection()
    if collection.count() == 0:
        ingest()

    results = collection.query(query_texts=[question], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches: list[dict[str, object]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )
    return matches


def ask_ollama(question: str, context: str, model: str = OLLAMA_MODEL) -> str:
    prompt = (
        "Answer using only the provided context. If the answer is not in the "
        "context, say you do not know. Cite the source and page when possible.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        if exc.code == 404:
            return (
                f"Ollama could not find the `{model}` model. Run "
                f"`ollama pull {model}` or choose an installed model in the sidebar."
            )
        return f"Ollama returned HTTP {exc.code}. {details}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return (
            "Could not connect to Ollama. Make sure Ollama is running and the "
            f"`{model}` model is available. Details: {reason}"
        )

    return data.get("response", "").strip() or "Ollama returned an empty response."


def _build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer using only the provided context. If the answer is not in "
                "the context, say you do not know. Cite the source and page when "
                "possible."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]


def ask_deepseek(question: str, context: str, model: str = DEEPSEEK_MODEL) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "DEEPSEEK_API_KEY is not set. Add your DeepSeek API key and try again."

    payload = {
        "model": model,
        "messages": _build_messages(question, context),
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        if exc.code == 401:
            return "DeepSeek rejected the API key. Check DEEPSEEK_API_KEY and try again."
        if exc.code == 404:
            return f"DeepSeek could not find the `{model}` model."
        return f"DeepSeek returned HTTP {exc.code}. {details}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"Could not connect to DeepSeek. Details: {reason}"

    choices = data.get("choices", [])
    if not choices:
        return "DeepSeek returned an empty response."
    message = choices[0].get("message", {})
    return message.get("content", "").strip() or "DeepSeek returned an empty response."


def ask_llm(
    question: str,
    context: str,
    provider: str = LLM_PROVIDER,
    model: str | None = None,
) -> str:
    provider = provider.lower()
    if provider == "ollama":
        return ask_ollama(question, context, model=model or OLLAMA_MODEL)
    if provider == "deepseek":
        return ask_deepseek(question, context, model=model or DEEPSEEK_MODEL)
    return f"Unsupported LLM provider `{provider}`. Choose one of: {', '.join(LLM_PROVIDERS)}."


def generate_content(
    question: str,
    content_type: str = "general",
    context_hint: str = "",
    provider: str = LLM_PROVIDER,
    model: str | None = None,
) -> str:
    """Generate content purely using the LLM without document context.
    
    Supports multiple content types: email, greeting, marketing,
    social_media, design_description, custom_template.
    """
    type_prompts = {
        "email": (
            "You are a professional email copywriter. Write polished, engaging, "
            "well-structured email content. Include a subject line suggestion, "
            "a warm greeting, clear body paragraphs, and a professional sign-off."
        ),
        "greeting": (
            "You are a creative greeting card writer. Craft warm, heartfelt, "
            "and memorable greeting messages. Make them personal, uplifting, "
            "and appropriate for the occasion."
        ),
        "marketing": (
            "You are an expert marketing copywriter. Create compelling, "
            "persuasive marketing content with strong hooks, clear value "
            "propositions, and powerful calls to action."
        ),
        "social_media": (
            "You are a social media content specialist. Write engaging, "
            "trend-aware captions and posts optimized for maximum engagement. "
            "Include relevant emoji, hashtag suggestions, and a compelling hook."
        ),
        "design_description": (
            "You are a creative director. Write vivid, detailed, and "
            "professional descriptions for visual designs, explaining the "
            "aesthetic, mood, color palette, and intended audience."
        ),
        "custom_template": (
            "You are a versatile content creator. Generate high-quality, "
            "structured content based on the user's template or brief. "
            "Be thorough and professional."
        ),
        "general": (
            "You are a knowledgeable and helpful AI assistant. Provide a "
            "thorough, accurate, and well-structured response to the user's request."
        ),
    }

    system_instruction = type_prompts.get(content_type, type_prompts["general"])

    context_section = ""
    if context_hint:
        context_section = (
            f"\n\nAdditional context / template information found in uploaded files:\n"
            f"{context_hint}\n\nUse this context to inform and enhance your generated content."
        )

    prompt_text = (
        f"{system_instruction}\n\n"
        f"User Request: {question}"
        f"{context_section}\n\n"
        "Provide a complete, professional response. Do not say you cannot help."
    )

    provider = provider.lower()
    if provider == "ollama":
        payload = {
            "model": model or OLLAMA_MODEL,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": 0.7},
        }
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as exc:
            return f"Content generation error: {exc}"

    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "DEEPSEEK_API_KEY is not set."
        payload = {
            "model": model or DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"{question}{context_section}"},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return "LLM returned empty response."
        except Exception as exc:
            return f"Content generation error: {exc}"

    return "Unsupported LLM provider."


def detect_content_type(question: str) -> str:
    """Detect the intended content type from the user's query."""
    q = question.lower()
    if any(k in q for k in ["email", "email content", "write email", "draft email"]):
        return "email"
    if any(k in q for k in ["greeting", "greeting card", "wish", "birthday", "anniversary"]):
        return "greeting"
    if any(k in q for k in ["marketing", "campaign", "advertisement", "ad copy", "promo"]):
        return "marketing"
    if any(k in q for k in ["social media", "caption", "post", "tweet", "instagram", "linkedin"]):
        return "social_media"
    if any(k in q for k in ["design description", "describe design", "design brief"]):
        return "design_description"
    if any(k in q for k in ["template", "fill template", "generate for", "based on template"]):
        return "custom_template"
    return "general"


def answer(
    question: str,
    n_results: int = 4,
    provider: str = LLM_PROVIDER,
    model: str | None = None,
) -> dict[str, object]:
    matches = retrieve(question, n_results=n_results)
    context = "\n\n".join(
        f"Source: {match['metadata']}\n{match['text']}" for match in matches
    )

    return {
        "answer": ask_llm(question, context, provider=provider, model=model),
        "sources": matches,
        "mode": "rag",
    }


def agentic_answer(
    question: str,
    n_results: int = 4,
    provider: str = LLM_PROVIDER,
    model: str | None = None,
) -> dict[str, object]:
    """Agentic RAG: intelligently choose between RAG retrieval, LLM generation, or hybrid.
    
    Decision logic:
      - ALL distances > threshold  → No relevant docs → Pure LLM generation
      - SOME distances <= threshold → Partial relevance → Hybrid (context + generation)
      - MOST distances <= threshold → Good relevance → Standard RAG
    """
    matches = retrieve(question, n_results=n_results)

    # Score relevance: distance <= threshold means the chunk IS relevant
    relevant = [m for m in matches if m["distance"] <= RELEVANCE_THRESHOLD]
    irrelevant = [m for m in matches if m["distance"] > RELEVANCE_THRESHOLD]

    content_type = detect_content_type(question)

    if not matches or len(relevant) == 0:
        # --- Pure Generation Mode ---
        generated = generate_content(question, content_type=content_type, provider=provider, model=model)
        return {
            "answer": generated,
            "sources": [],
            "mode": "generated",
            "content_type": content_type,
            "decision": "No relevant documents found. Content was generated by the LLM.",
        }

    elif len(relevant) < len(matches) / 2:
        # --- Hybrid Mode: partial context + generation ---
        context_hint = "\n\n".join(
            f"Source: {m['metadata']}\n{m['text']}" for m in relevant
        )
        generated = generate_content(
            question, content_type=content_type, context_hint=context_hint,
            provider=provider, model=model,
        )
        return {
            "answer": generated,
            "sources": relevant,
            "mode": "hybrid",
            "content_type": content_type,
            "decision": (
                f"Partial match found ({len(relevant)}/{len(matches)} chunks relevant). "
                "Combined document context with LLM generation."
            ),
        }

    else:
        # --- Standard RAG Mode ---
        context = "\n\n".join(
            f"Source: {m['metadata']}\n{m['text']}" for m in matches
        )
        return {
            "answer": ask_llm(question, context, provider=provider, model=model),
            "sources": matches,
            "mode": "rag",
            "content_type": content_type,
            "decision": (
                f"Found {len(relevant)}/{len(matches)} highly relevant chunks. "
                "Used document retrieval (RAG)."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Vector RAG demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index files from data/")
    ingest_parser.add_argument("--reset", action="store_true", help="Rebuild collection")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--results", type=int, default=4)
    ask_parser.add_argument(
        "--provider",
        choices=LLM_PROVIDERS,
        default=LLM_PROVIDER if LLM_PROVIDER in LLM_PROVIDERS else "ollama",
        help="LLM provider",
    )
    ask_parser.add_argument("--model", help="Provider model name")

    args = parser.parse_args()
    if args.command == "ingest":
        count = ingest(reset=args.reset)
        print(f"Indexed {count} chunks into {CHROMA_DIR / COLLECTION_NAME}.")
    elif args.command == "ask":
        result = answer(
            args.question,
            n_results=args.results,
            provider=args.provider,
            model=args.model,
        )
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(source["metadata"])


if __name__ == "__main__":
    main()
