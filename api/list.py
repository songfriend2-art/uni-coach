"""
업로드된 대학 목록 반환
"""
import os
import json
from http.server import BaseHTTPRequestHandler
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            result = sb.table("documents") \
                .select("university, filename") \
                .execute()

            seen = {}
            for row in (result.data or []):
                univ = row["university"]
                fname = row["filename"]
                if univ not in seen:
                    seen[univ] = fname

            universities = [
                {"university": k, "filename": v}
                for k, v in sorted(seen.items())
            ]

            self._json(200, {"universities": universities})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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
