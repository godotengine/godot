import http.server
import socketserver


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".pck"):
            return "application/octet-stream"
        return super().guess_type(path)

    def send_response_only(self, code, message=None):
        super().send_response_only(code, message)
        # Disable caching for .wasm and .js
        if self.path.endswith((".wasm", ".js", ".pck")):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")


with socketserver.TCPServer(("", 8000), Handler) as httpd:
    print("Serving at http://localhost:8000")
    httpd.serve_forever()
