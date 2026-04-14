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
จากเนื้อหาด้านล่าง กรุณาสรุปเป็นภาษาที่ชัดเจน กระชับ ไม่เกิน 3-4 ประโยค
เน้นสาระสำคัญที่เกี่ยวข้องกับคำค้นหา
ขอเป็นภาษาไทย: "{query}"

URL: {url}
เนื้อหา:{content}

สรุป:
"""

async def _summarize_ollama(prompt: str) -> str:
    """
    ใช้ Ollama local server — compatible กับ OpenAI SDK
    เพียงแค่เปลี่ยน base_url และไม่ต้องใส่ api_key จริง
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",               # Ollama ไม่ต้องการ key จริง ใส่อะไรก็ได้
    )
    resp = await client.chat.completions.create(
        model=LLM_MODEL,                # เช่น "llama3.2", "gemma3", "typhoon2"
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


async def _summarize_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = await client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


async def _summarize_openai(prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


async def summarize_one(query: str, url: str, content: str) -> str:
    """สรุปเนื้อหาหน้าเดียว"""
    if not content.strip():
        return "ไม่สามารถดึงเนื้อหาได้"

    prompt = SUMMARIZE_PROMPT.format(
        query=query,
        url=url,
        content=content[:2500],  # ป้องกัน token overflow
    )

    if LLM_PROVIDER == "anthropic":
        return await _summarize_anthropic(prompt)
    elif LLM_PROVIDER == "ollama":
        return await _summarize_ollama(prompt)
    else:
        return await _summarize_openai(prompt)


async def summarize_all(query: str, pages: list) -> list:
    """
    สรุปทุกหน้าพร้อมกัน (concurrent)
    pages: list of WebResult
    """
    # จำกัด concurrency ไม่ให้ spam API
    semaphore = asyncio.Semaphore(5)

    async def _bounded(page):
        async with semaphore:
            page.summary = await summarize_one(query, page.url, page.content)
        return page

    return await asyncio.gather(*[_bounded(p) for p in pages])

