"""
embedder.py — สร้าง embedding vectors จากข้อความ
รองรับ:
  - OpenAI text-embedding-3-small  → 1536 dims
  - Ollama nomic-embed-text        → 768  dims
  - Ollama mxbai-embed-large       → 1024 dims
  - Ollama bge-m3 (multilingual)   → 1024 dims  ← แนะนำถ้าข้อความมีภาษาไทย

⚠️  ถ้าเปลี่ยน embedding model ต้องล้าง DB แล้วสร้างใหม่
    เพราะ dimension ไม่เท่ากันจะ insert ไม่ได้
"""
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()  # "openai" | "ollama"
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Map model → dimension (ใช้ตรวจสอบก่อน insert)
MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text":       768,
    "mxbai-embed-large":      1024,
    "bge-m3":                 1024,
    "all-minilm":             384,
}


def get_embedding_dim() -> int:
    """คืน dimension ของ model ที่เลือก"""
    return MODEL_DIMS.get(EMBEDDING_MODEL, 1536)


def _openai_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _ollama_client():
    """Ollama เปิด OpenAI-compatible endpoint ที่ /v1"""
    from openai import AsyncOpenAI
    return AsyncOpenAI(  
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )


async def embed_text(text: str) -> list[float]:
    """แปลง text → vector"""
    text = text.replace("\n", " ").strip()[:8000]
    client = _ollama_client() if EMBEDDING_PROVIDER == "ollama" else _openai_client()
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """batch embed — Ollama ไม่รองรับ batch จริง จึง loop แทน"""
    cleaned = [t.replace("\n", " ").strip()[:8000] for t in texts]

    if EMBEDDING_PROVIDER == "ollama":
        # Ollama /v1/embeddings รับ list ได้ แต่บาง model อาจ OOM
        # ปลอดภัยสุดคือส่งทีละอัน
        import asyncio
        client = _ollama_client()
        results = []
        for text in cleaned:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            results.append(resp.data[0].embedding)
        return results
    else:
        client = _openai_client()
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=cleaned)
        return [item.embedding for item in response.data]