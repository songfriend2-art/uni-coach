"""
PDF 업로드 → 텍스트 추출 → 청크 분할 → 임베딩 → Supabase 저장
"""
import os
import json
import re
import tempfile
import hashlib
import math
from http.server import BaseHTTPRequestHandler
import pdfplumber
import google.generativeai as genai
from supabase import create_client

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100


def extract_text(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
            for table in page.extract_tables():
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cells))
                if rows:
                    texts.append("\n".join(rows))
    return "\n\n".join(texts)


def guess_university(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r"(20\d{2}|입학|전형|안내|모집|요강|대학교?)", "", name)
    name = re.sub(r"[_\-\s]+", " ", name).strip()
    return name or filename


def split_chunks(text: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_embedding(text: str) -> list:
    """Gemini 임베딩 (768차원 → 1536 패딩)"""
    genai.configure(api_key=GEMINI_KEY)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text[:2000],
        task_type="retrieval_document"
    )
    emb = result["embedding"]
    if len(emb) < 1536:
        emb = emb + [0.0] * (1536 - len(emb))
    return emb[:1536]


def get_embedding_fallback(text: str) -> list:
    words = re.findall(r"[가-힣a-zA-Z]+", text.lower())
    vec = [0.0] * 1536
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % 1536] += 1.0
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


def parse_multipart(body: bytes, content_type: str):
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[9:].strip('"')
    if not boundary:
        return None, None, None

    sep = ("--" + boundary).encode()
    parts = body.split(sep)
    filename = file_data = university = None

    for p in parts:
        if b"Content-Disposition" not in p:
            continue
        header, _, content = p.partition(b"\r\n\r\n")
        header_str = header.decode(errors="ignore")
        content = content.rstrip(b"\r\n--")

        if 'name="university"' in header_str:
            university = content.decode(errors="ignore").strip()
        elif 'name="file"' in header_str:
            m = re.search(r'filename="([^"]+)"', header_str)
            if m:
                filename = m.group(1)
                file_data = content

    return filename, file_data, university


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        filename, file_data, university_override = parse_multipart(body, content_type)

        if not filename or not file_data:
            self._json(400, {"error": "파일이 없습니다"})
            return

        if not filename.lower().endswith(".pdf"):
            self._json(400, {"error": "PDF 파일만 지원합니다"})
            return

        university = university_override or guess_university(filename)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            raw_text = extract_text(tmp_path)
            if not raw_text.strip():
                self._json(400, {"error": "PDF에서 텍스트를 추출할 수 없습니다"})
                return

            chunks = split_chunks(raw_text)
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            sb.table("documents").delete().eq("filename", filename).execute()

            saved = 0
            for chunk in chunks:
                try:
                    embedding = get_embedding(chunk)
                except Exception:
                    embedding = get_embedding_fallback(chunk)

                sb.table("documents").insert({
                    "university": university,
                    "filename": filename,
                    "content": chunk,
                    "embedding": embedding
                }).execute()
                saved += 1

            os.unlink(tmp_path)
            self._json(200, {
                "success": True,
                "university": university,
                "filename": filename,
                "chunks": saved
            })

        except Exception as e:
            try:
                os.unlink(tmp_path)
            except:
                pass
            self._json(500, {"error": str(e)})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
