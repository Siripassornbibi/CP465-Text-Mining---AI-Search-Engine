"""
searcher.py — ค้นหาเว็บด้วย DuckDuckGo และ scrape เนื้อหา
"""
import asyncio
import os
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "3000"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class WebResult:
    url: str
    title: str
    snippet: str          # snippet จาก search engine
    content: str = ""     # เนื้อหาจากการ scrape
    summary: str = ""     # สรุปจาก LLM


def search_web(query: str, max_results: int = MAX_RESULTS) -> list[WebResult]:
    """ค้นหาด้วย DuckDuckGo (sync) — คืน list ของ WebResult"""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                WebResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                )
            )
    return results


def _extract_text(html: str) -> str:
    """ดึงเฉพาะข้อความที่อ่านได้จาก HTML"""
    soup = BeautifulSoup(html, "html.parser")

    # ลบ script, style, nav, footer ทิ้ง
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # ลอง main content ก่อน
    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # ทำความสะอาด whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:MAX_CONTENT_LENGTH]


async def scrape_url(client: httpx.AsyncClient, url: str) -> str:
    """ดึงเนื้อหาจาก URL เดียว (async)"""
    try:
        resp = await client.get(url, headers=HEADERS, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        return _extract_text(resp.text)
    except Exception as exc:
        return ""  # ถ้าล้มเหลว คืน string ว่าง


async def scrape_all(results: list[WebResult]) -> list[WebResult]:
    """scrape ทุก URL พร้อมกัน (concurrent)"""
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(client, r.url) for r in results]
        contents = await asyncio.gather(*tasks)
    for result, content in zip(results, contents):
        result.content = content
    return results
