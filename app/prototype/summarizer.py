"""
summarizer.py — สรุปเนื้อหาเว็บด้วย LLM
หลักๆเน้น Ollama
+ ตัวอย่าง
รองรับทั้ง Anthropic (Claude) และ OpenAI (GPT-4)
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


SUMMARIZE_PROMPT = """\
คุณเป็น AI ที่ช่วยสรุปเนื้อหาเว็บไซต์
จากเนื้อหาที่คัดมาด้านล่าง (แบ่งเป็นส่วนๆ ที่เกี่ยวข้องกับคำค้นหา)
กรุณาสรุปเป็นภาษาที่ชัดเจน กระชับ ไม่เกิน 4-5 ประโยค
เน้นสาระสำคัญที่เกี่ยวข้องกับคำค้นหา: "{query}"

URL: {url}

เนื้อหาที่คัดมา:
{context}

สรุป:"""

async def _call_llm(prompt: str) -> str:
    """เรียก LLM ตาม provider ที่ตั้งค่าไว้"""
    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = await client.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
 
    elif LLM_PROVIDER == "ollama":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
 
    else:  # openai
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
 
 
async def summarize_one(
    query: str,
    url: str,
    content: str,
    query_emb: list[float] | None = None,
) -> str:
    """
    สรุปเนื้อหาหน้าเดียวด้วย RAG
    1. แบ่ง content เป็น chunks
    2. retrieve เฉพาะ chunk ที่เกี่ยวข้องกับ query
    3. ส่ง context ที่คัดแล้วให้ LLM สรุป
    """
    if not content.strip():
        return "ไม่สามารถดึงเนื้อหาได้"
 
    # RAG: ดึงเฉพาะ chunk ที่เกี่ยวข้อง
    from app.prototype.chunker import get_relevant_context
    context = await get_relevant_context(query, query_emb or [], content)
 
    prompt = SUMMARIZE_PROMPT.format(
        query=query,
        url=url,
        context=context,
    )
 
    return await _call_llm(prompt)
 
 
async def summarize_all(
    query: str,
    pages: list,
    query_emb: list[float] | None = None,
) -> list:
    """
    สรุปทุกหน้าพร้อมกัน พร้อมส่ง query_emb สำหรับ RAG retrieval
    """
    concurrency = 2 if LLM_PROVIDER == "ollama" else 5
    semaphore = asyncio.Semaphore(concurrency)
 
    async def _bounded(page):
        async with semaphore:
            page.summary = await summarize_one(
                query, page.url, page.content, query_emb
            )
        return page
 
    return await asyncio.gather(*[_bounded(p) for p in pages])
 