"""
사용자 질문 → 임베딩 → Supabase 벡터 검색 → Gemini 답변 생성
"""
import os
import json
import re
import hashlib
import math
from http.server import BaseHTTPRequestHandler
import google.generativeai as genai
from supabase import create_client

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_KEY    = os.environ["GEMINI_API_KEY"]

SYSTEM_PROMPT = """당신은 대한민국 최고의 대학입시 전문 컨설턴트입니다.
학생부종합·교과·수능·논술 등 모든 전형에 정통하며 15년 이상의 상담 경험을 갖고 있습니다.

## 답변 원칙
- [참고 자료]가 제공된 경우 해당 내용을 최우선으로 활용하고 출처(대학명, 전형명)를 명시
- 참고 자료에 없는 내용은 일반 지식으로 보완하되 "일반 정보 기준" 명시
- 추천 대학은 상위/중위/안정권으로 구분
- 선택과목 조합 구체적 명시 (예: 미적분+물리학Ⅱ+화학Ⅱ)
- 생기부 활동 구체적 예시 제공
- 수능 최저기준, 내신 반영 비율 등 실제 전형 정보 포함
- 불확실한 정보는 "확인 필요" 명시
- 한국어로 친근하고 명확하게, 구조화된 형식으로 답변
- 2024-2025학년도 입시 기준"""


def get_embedding(text: str) -> list:
    """Gemini 임베딩 생성"""
    genai.configure(api_key=GEMINI_KEY)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text[:2000],
        task_type="retrieval_query"
    )
    # Gemini 임베딩은 768차원 → 1536에 맞게 패딩
    emb = result["embedding"]
    if len(emb) < 1536:
        emb = emb + [0.0] * (1536 - len(emb))
    return emb[:1536]


def get_embedding_fallback(text: str) -> list:
    """폴백: 키워드 기반 sparse 벡터"""
    words = re.findall(r"[가-힣a-zA-Z]+", text.lower())
    vec = [0.0] * 1536
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % 1536] += 1.0
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


def search_docs(query: str, top_k: int = 6) -> list:
    """Supabase 벡터 검색"""
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    try:
        embedding = get_embedding(query)
    except Exception:
        embedding = get_embedding_fallback(query)

    result = sb.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": top_k
    }).execute()
    return result.data or []


def build_context(rows: list) -> str:
    if not rows:
        return ""
    parts = []
    for r in rows:
        univ = r.get("university", "")
        sim  = r.get("similarity", 0)
        content = r.get("content", "")
        parts.append(f"[{univ} | 유사도 {sim:.2f}]\n{content}")
    return "\n\n---\n\n".join(parts)


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))

        messages = body.get("messages", [])
        profile  = body.get("profile", {})

        if not messages:
            self._json(400, {"error": "메시지가 없습니다"})
            return

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # 벡터 검색
        docs = search_docs(last_user_msg)
        context = build_context(docs)

        # 시스템 프롬프트 구성
        system = SYSTEM_PROMPT
        if context:
            system += f"\n\n## 참고 자료 (입학전형 파일 기반)\n{context}"
        if profile:
            parts = []
            if profile.get("major"):    parts.append(f"희망학과: {profile['major']}")
            if profile.get("year"):     parts.append(f"학년: {profile['year']}")
            if profile.get("naesin"):   parts.append(f"내신: {profile['naesin']}등급")
            if profile.get("mock"):     parts.append(f"모의고사: {profile['mock']}%")
            if profile.get("subjects"): parts.append(f"강점과목: {profile['subjects']}")
            if profile.get("types"):    parts.append(f"선호전형: {profile['types']}")
            if parts:
                system += "\n\n## 학생 프로필\n" + "\n".join(parts)

        # Gemini API 호출
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system
        )

        # 대화 히스토리 변환 (Gemini 형식)
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

        chat = model.start_chat(history=history)
        response = chat.send_message(last_user_msg)
        reply = response.text

        sources = list({r.get("university", "") for r in docs if r.get("university")})

        self._json(200, {
            "reply": reply,
            "sources": sources,
            "context_used": bool(context)
        })

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
