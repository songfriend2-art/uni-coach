"""
scripts/batch_upload.py
로컬 PDF 폴더 전체를 Supabase에 일괄 업로드

사용법:
  pip install -r requirements.txt
  python scripts/batch_upload.py ./pdfs

환경변수 설정 (실행 전):
  set SUPABASE_URL=https://rsfxtaotgpzorotmcogb.supabase.co
  set SUPABASE_SERVICE_KEY=your_secret_key
  set GEMINI_API_KEY=your_gemini_key
"""
import os
import sys
import re
import math
import hashlib
import tempfile

import pdfplumber
import google.generativeai as genai
from supabase import create_client

SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100


def extract_text(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
            for table in page.extract_tables():
                rows = [" | ".join(str(c).strip() if c else "" for c in row) for row in table]
                if rows:
                    texts.append("\n".join(rows))
    return "\n\n".join(texts)


def guess_university(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"(20\d{2}|입학|전형|안내|모집|요강|대학교?)", "", name)
    name = re.sub(r"[_\-\s]+", " ", name).strip()
    return name or filename


def split_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_embedding(text):
    if GEMINI_KEY:
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
    else:
        words = re.findall(r"[가-힣a-zA-Z]+", text.lower())
        vec = [0.0] * 1536
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vec[h % 1536] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]


def upload_pdf(sb, filepath):
    filename = os.path.basename(filepath)
    university = guess_university(filename)
    print(f"  [{university}] 텍스트 추출 중...", end=" ", flush=True)

    raw = extract_text(filepath)
    if not raw.strip():
        print("❌ 텍스트 없음 (스캔 PDF)")
        return 0

    chunks = split_chunks(raw)
    print(f"{len(chunks)}개 청크", flush=True)

    sb.table("documents").delete().eq("filename", filename).execute()

    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        sb.table("documents").insert({
            "university": university,
            "filename": filename,
            "content": chunk,
            "embedding": emb
        }).execute()
        print(f"    {i+1}/{len(chunks)} 저장중...", end="\r", flush=True)

    print(f"    ✅ {len(chunks)}개 청크 완료          ")
    return len(chunks)


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/batch_upload.py <PDF폴더경로>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"폴더를 찾을 수 없습니다: {folder}")
        sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("환경변수를 먼저 설정해주세요:")
        print("  set SUPABASE_URL=...")
        print("  set SUPABASE_SERVICE_KEY=...")
        print("  set GEMINI_API_KEY=...")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    print(f"\n총 {len(pdfs)}개 PDF 처리 시작\n")

    total = 0
    for i, fname in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {fname}")
        try:
            total += upload_pdf(sb, os.path.join(folder, fname))
        except Exception as e:
            print(f"  ❌ 오류: {e}")

    print(f"\n✅ 완료! {len(pdfs)}개 파일, {total}개 청크 저장")


if __name__ == "__main__":
    main()
