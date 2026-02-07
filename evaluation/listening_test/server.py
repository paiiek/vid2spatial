#!/usr/bin/env python3
"""
Simple HTTP server for the Vid2Spatial perceptual evaluation.

Serves static files (HTML, audio, video) and handles response saving.

Usage:
    cd evaluation/listening_test
    python server.py [--port 8080]

Then open http://localhost:8080 in a browser.
"""
import os
import sys
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime


RESPONSES_DIR = Path(__file__).parent / "responses"


class EvalHandler(SimpleHTTPRequestHandler):
    """HTTP handler with response saving API."""

    def do_POST(self):
        if self.path == '/api/save':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                self._save_response(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _save_response(self, data):
        RESPONSES_DIR.mkdir(exist_ok=True)
        pid = data.get('participant_id', 'unknown')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{pid}_{ts}.json"
        path = RESPONSES_DIR / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[saved] {path}")

    def end_headers(self):
        # CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress routine GET logs, show only important ones
        if 'POST' in str(args) or 'ERROR' in str(args):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description='Vid2Spatial listening test server')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    # Change to the listening_test directory
    os.chdir(Path(__file__).parent)

    server = HTTPServer((args.host, args.port), EvalHandler)
    print(f"\n  Vid2Spatial Perceptual Evaluation Server")
    print(f"  ========================================")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  Responses saved to: {RESPONSES_DIR}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stopped]")
        server.server_close()


if __name__ == "__main__":
    main()
