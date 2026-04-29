"""
cli.py — ทดสอบ search engine ผ่าน terminal
รัน: python cli.py "python async tutorial"
"""
import asyncio
import sys
import textwrap
from app.prototype.engine import get_engine


async def main(query: str):
    engine = await get_engine()
    results = await engine.search(query)

    print(f"\n{'─'*60}")
    print(f"ผลการค้นหา: {query!r}  ({len(results)} รายการ)")
    print(f"{'─'*60}\n")

    for i, r in enumerate(results, 1):
        cache_tag = "🗄️ [cache]" if r.from_cache else "🌐 [live]"
        print(f"{i:2d}. {cache_tag} {r.title}")
        print(f"    🔗 {r.url}")
        summary_wrapped = textwrap.fill(r.summary, width=70, initial_indent="    📝 ", subsequent_indent="       ")
        print(summary_wrapped)
        print()

    await engine.close()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "python asyncio tutorial"
    asyncio.run(main(q))
