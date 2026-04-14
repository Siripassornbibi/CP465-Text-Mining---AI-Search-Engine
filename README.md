# 🔍 AI Search Engine — Python + PostgreSQL pgvector

ระบบค้นหาเว็บที่ใช้ LLM สรุปผลลัพธ์ และ pgvector เป็น vector database สำหรับ semantic cache

## Architecture

```
User Query
    │
    ▼
[Embedding]  ← OpenAI text-embedding-3-small
    │
    ▼
[pgvector Cache Lookup]  ← cosine similarity > 0.75 → คืน cache
    │ cache miss
    ▼
[DuckDuckGo Search]  → 10 URLs 🦖 ตอนนี้ลองแค่ 1 ยังนาน
    │
    ▼
[Concurrent Scraper]  ← httpx + BeautifulSoup
    │
    ▼
[LLM Summarizer]  ← Claude / GPT-4 (concurrent, semaphore=5) 🦖 ตอนนี้ใช้ ollama 3.2
    │
    ▼
[Store to pgvector]  ← embed content + store cache 🦖 ตอนนี้ใช้ bge-m3 
    │
    ▼
Results (10 URLs + summaries)
```

## การติดตั้ง

### 1. เริ่ม PostgreSQL ด้วย Docker => 🦖 ใช้ brew แทน
```bash
docker compose up -d
```

### 2. ติดตั้ง Python packages >> 🦖 ใช้ python 3.12
```bash
python -m venv venv  >> 🦖 ใช้ python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 🦖 ตั้งค่า environment
```bash
cp .env.example .env
# แก้ไข .env ใส่ API keys 🦖 แก้แค่พวก POSTGRES_USER/POSTGRES_PASSWORD
```

ค่าที่ต้องใส่:
| ตัวแปร | คำอธิบาย |
|--------|---------|
| `OPENAI_API_KEY` | ใช้สำหรับ embedding (จำเป็น) |
| `ANTHROPIC_API_KEY` | ถ้าใช้ Claude เป็น LLM |
| `LLM_PROVIDER` | `anthropic` หรือ `openai` |
| `LLM_MODEL` | เช่น `claude-opus-4-5` หรือ `gpt-4o` |

## การใช้งาน

### CLI (ทดสอบเร็ว)
```bash
python cli.py "machine learning tutorial"
python cli.py "วิธีทำ pad thai"
```

🦖 ลองมาถึงตรงนี้ปรับแก้ version library ให้รันผ่านแล้ว

### API Server
```bash
uvicorn api:app --reload
```

เปิด http://localhost:8000/docs เพื่อดู Swagger UI

#### ตัวอย่าง API call
```bash
# GET
curl "http://localhost:8000/search?q=python+asyncio"

# POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does pgvector work"}'
```

#### ตัวอย่าง Response
```json
{
  "query": "python asyncio tutorial",
  "total": 10,
  "elapsed_ms": 4523.1,
  "from_cache": false,
  "results": [
    {
      "rank": 1,
      "url": "https://docs.python.org/3/library/asyncio.html",
      "title": "asyncio — Asynchronous I/O",
      "snippet": "asyncio is a library to write concurrent code...",
      "summary": "หน้าเอกสารทางการของ Python สำหรับ asyncio...",
      "from_cache": false
    }
  ]
}
```

## โครงสร้างไฟล์

```
ai_search/
├── api.py           ← FastAPI REST server
├── cli.py           ← CLI สำหรับทดสอบ
├── engine.py        ← Orchestrator หลัก
├── db.py            ← PostgreSQL + pgvector
├── embedder.py      ← OpenAI embedding
├── searcher.py      ← DuckDuckGo + scraper
├── summarizer.py    ← LLM summarization
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Tuning

| Parameter | ไฟล์ | คำอธิบาย |
|-----------|------|---------|
| `SIMILARITY_THRESHOLD` | `.env` | ยิ่งสูง cache hit น้อยลงแต่แม่นขึ้น (0.0–1.0) |
| `MAX_CONTENT_LENGTH` | `.env` | ตัดเนื้อหาเว็บกี่ตัวอักษร |
| `MAX_RESULTS` | `.env` | จำนวนผลลัพธ์ |
| `m`, `ef_construction` | `db.py` | HNSW index parameters |
