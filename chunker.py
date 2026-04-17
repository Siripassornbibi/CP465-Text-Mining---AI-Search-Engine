"""
chunker.py — แบ่งเนื้อหาเว็บเป็น chunks พร้อม overlap
และ retrieve chunks ที่เกี่ยวข้องกับ query ผ่าน cosine similarity
"""
import os
import math
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))      # ตัวอักษรต่อ chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))   # overlap ระหว่าง chunk
TOP_K_CHUNKS  = int(os.getenv("TOP_K_CHUNKS", "4"))      # chunk ที่ส่งให้ LLM


@dataclass
class Chunk:
    index: int
    text: str
    embedding: list[float] | None = None


def split_chunks(text: str) -> list[Chunk]:
    """
    Sliding window chunker — แบ่งข้อความโดยพยายามตัดที่ขอบประโยค
    เพื่อไม่ให้ตัดกลางคำ
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # พยายามตัดที่ขอบประโยคหรือบรรทัด ไม่ตัดกลางคำ
        if end < len(text):
            # หา newline หรือ ". " ที่ใกล้สุดก่อน end
            cut = text.rfind("\n", start, end)
            if cut == -1 or cut < start + step // 2:
                cut = text.rfind(". ", start, end)
            if cut != -1 and cut > start + step // 2:
                end = cut + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(index=idx, text=chunk_text))
            idx += 1

        start += step

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """คำนวณ cosine similarity แบบ in-memory (ไม่ต้องใช้ DB)"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Embed ทุก chunk (ใช้ embedder ที่มีอยู่แล้ว)"""
    from embedder import embed_texts
    texts = [c.text for c in chunks]
    embeddings = await embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    return chunks


async def retrieve_relevant_chunks(
    query_emb: list[float],
    chunks: list[Chunk],
    top_k: int = TOP_K_CHUNKS,
) -> list[Chunk]:
    """
    คัดเลือก chunk ที่เกี่ยวข้องกับ query มากที่สุด
    เรียงตาม cosine similarity แล้วเอา top_k อัน
    """
    scored = [
        (cosine_similarity(query_emb, c.embedding), c)
        for c in chunks
        if c.embedding is not None
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # เอา top_k แล้วเรียง index กลับ เพื่อให้เนื้อหาอ่านต่อเนื่อง
    top = [c for _, c in scored[:top_k]]
    top.sort(key=lambda c: c.index)
    return top


async def get_relevant_context(
    query: str,
    query_emb: list[float],
    content: str,
) -> str:
    """
    Pipeline ครบ:
    content → chunks → embed → retrieve top-K → join เป็น context string
    """
    chunks = split_chunks(content)
    if not chunks:
        return content[:2500]  # fallback ถ้าสั้นเกินแบ่งไม่ได้

    # ถ้า chunk น้อยกว่า TOP_K ไม่ต้อง embed ก็ได้ ส่งทั้งหมดเลย
    if len(chunks) <= TOP_K_CHUNKS:
        return "\n\n---\n\n".join(c.text for c in chunks)

    chunks = await embed_chunks(chunks)
    relevant = await retrieve_relevant_chunks(query_emb, chunks)

    context_parts = []
    for c in relevant:
        context_parts.append(f"[ส่วนที่ {c.index + 1}]\n{c.text}")

    return "\n\n---\n\n".join(context_parts)