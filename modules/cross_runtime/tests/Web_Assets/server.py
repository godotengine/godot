import http.server
import socketserver


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/log":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            print(f"BROWSER_LOG: {post_data.decode('utf-8')}")
            self.send_response(200)
            self.end_headers()

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


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", 8000), Handler) as httpd:
    print("Server ready. Listening for logs at http://localhost:8000")
    httpd.serve_forever()
