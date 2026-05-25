"""
사용자 질문 → Supabase 벡터 검색 → Groq 스트리밍 답변
한글 UTF-8 버퍼 처리 수정
"""
import os
import json
import re
import hashlib
import math
import http.client
from http.server import BaseHTTPRequestHandler
from supabase import create_client

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_KEY      = os.environ["GROQ_API_KEY"]

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
- 2028학년도 입시 기준"""


def get_embedding_fallback(text: str) -> list:
    words = re.findall(r"[가-힣a-zA-Z]+", text.lower())
    vec = [0.0] * 1536
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % 1536] += 1.0
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


def search_docs(query: str, university_filter: str = None, top_k: int = 6) -> list:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    embedding = get_embedding_fallback(query)
    result = sb.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": top_k
    }).execute()
    rows = result.data or []
    if university_filter:
        rows = [r for r in rows if university_filter in r.get("university", "")]
    return rows


def build_context(rows: list) -> str:
    if not rows:
        return ""
    parts = []
    for r in rows:
        univ    = r.get("university", "")
        sim     = r.get("similarity", 0)
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

        messages          = body.get("messages", [])
        university_filter = body.get("university", None)
        profile           = body.get("profile", {})

        if not messages:
            self._stream_error("메시지가 없습니다")
            return

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # 벡터 검색
        docs    = search_docs(last_user_msg, university_filter)
        context = build_context(docs)
        sources = list({r.get("university", "") for r in docs if r.get("university")})

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

        # 스트리밍 헤더 전송
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # sources 먼저 전송
        self._send_event("sources", json.dumps(sources, ensure_ascii=False))

        # Groq 호출
        try:
            groq_messages = [{"role": "system", "content": system}]
            for m in messages[-10:]:
                groq_messages.append({"role": m["role"], "content": m["content"]})

            payload = json.dumps({
                "model": "llama-3.3-70b-versatile",
                "messages": groq_messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 2000
            })

            conn = http.client.HTTPSConnection("api.groq.com")
            conn.request(
                "POST",
                "/openai/v1/chat/completions",
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + GROQ_KEY.strip(),
                    "User-Agent": "python-httplib"
                }
            )
            resp = conn.getresponse()

            if resp.status != 200:
                err = resp.read().decode("utf-8", errors="replace")
                self._send_event("error", f"Groq 오류 {resp.status}: {err}")
                return

            # 바이트 버퍼로 한글 잘림 방지
            byte_buffer = b""
            line_buffer = ""

            while True:
                chunk = resp.read(256)
                if not chunk:
                    break

                byte_buffer += chunk
                # UTF-8 안전 디코딩 (잘린 멀티바이트 보호)
                try:
                    decoded = byte_buffer.decode("utf-8")
                    byte_buffer = b""
                except UnicodeDecodeError:
                    # 마지막 바이트가 잘렸을 경우 다음 청크를 기다림
                    try:
                        decoded = byte_buffer[:-1].decode("utf-8")
                        byte_buffer = byte_buffer[-1:]
                    except UnicodeDecodeError:
                        try:
                            decoded = byte_buffer[:-2].decode("utf-8")
                            byte_buffer = byte_buffer[-2:]
                        except UnicodeDecodeError:
                            try:
                                decoded = byte_buffer[:-3].decode("utf-8")
                                byte_buffer = byte_buffer[-3:]
                            except UnicodeDecodeError:
                                continue

                line_buffer += decoded

                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        self._send_event("done", "")
                        conn.close()
                        return
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            self._send_event("delta", json.dumps({"t": delta}, ensure_ascii=False))
                    except Exception:
                        continue

            self._send_event("done", "")
            conn.close()

        except Exception as e:
            self._send_event("error", str(e))

    def _send_event(self, event: str, data: str):
        try:
            msg = f"event: {event}\ndata: {data}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
        except Exception:
            pass

    def _stream_error(self, msg: str):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        self._send_event("error", msg)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *args):
        pass
