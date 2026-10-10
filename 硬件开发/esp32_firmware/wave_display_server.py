"""Separate display frontend so page fixes never interrupt a live capture."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from experiment_wave_viewer import HTML

class DisplayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body, mime = HTML.encode(), 'text/html; charset=utf-8'
        elif self.path in ('/data', '/download'):
            with urlopen('http://127.0.0.1:8011' + self.path, timeout=3) as response:
                body, mime = response.read(), response.headers['Content-Type']
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        if self.path == '/download':
            self.send_header('Content-Disposition', 'attachment; filename="signals.csv"')
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == '__main__':
    print('DISPLAY http://127.0.0.1:8012/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8012), DisplayHandler).serve_forever()
