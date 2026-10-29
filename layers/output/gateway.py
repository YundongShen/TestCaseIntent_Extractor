import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional

from layers.output.renderers import EXTENSIONS, MEDIA_TYPES, RENDERERS, render

Handler = Callable[[Dict], object]


def passthrough(payload: Dict):
    return payload.get("content", "")


class Gateway:
    def __init__(self, handler: Optional[Handler] = None):
        self.handler = handler or passthrough

    def run(self, payload: Dict, fmt="markdown") -> bytes:
        return render(self.handler(payload), fmt)

    def serve(self, host="127.0.0.1", port=8000):
        gateway = self

        class RequestHandler(BaseHTTPRequestHandler):
            def _send(self, code, body, media="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", media)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, json.dumps({"status": "ok", "formats": sorted(RENDERERS)}).encode())
                else:
                    self._send(404, b'{"error": "not found"}')

            def do_POST(self):
                if self.path not in ("/run", "/render"):
                    self._send(404, b'{"error": "not found"}')
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    fmt = payload.get("format", "markdown")
                    content = payload.get("content", "") if self.path == "/render" else gateway.handler(payload)
                    self._send(200, render(content, fmt), MEDIA_TYPES.get(fmt, "application/octet-stream"))
                except Exception as e:
                    self._send(400, json.dumps({"error": str(e)}).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer((host, port), RequestHandler)
        return server

    def ci(self, inputs, fmt="markdown", out_dir="build/onboarding", report_path=None):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for path in inputs:
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                target = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "." + EXTENSIONS[fmt])
                with open(target, "wb") as f:
                    f.write(self.run({"content": content, "source": path}, fmt))
                results.append({"input": path, "output": target, "status": "ok"})
            except Exception as e:
                results.append({"input": path, "status": "failed", "error": str(e)})
        report = {"format": fmt, "total": len(results),
                  "failed": sum(r["status"] != "ok" for r in results), "results": results}
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return report


def cli(argv=None, gateway: Optional[Gateway] = None):
    gateway = gateway or Gateway()
    parser = argparse.ArgumentParser(prog="onboarding-gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render")
    p_render.add_argument("input")
    p_render.add_argument("--format", default="markdown", choices=sorted(RENDERERS))
    p_render.add_argument("--output")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_ci = sub.add_parser("ci")
    p_ci.add_argument("inputs", nargs="+")
    p_ci.add_argument("--format", default="markdown", choices=sorted(RENDERERS))
    p_ci.add_argument("--out-dir", default="build/onboarding")
    p_ci.add_argument("--report")

    args = parser.parse_args(argv)
    if args.command == "render":
        with open(args.input, encoding="utf-8") as f:
            data = gateway.run({"content": f.read(), "source": args.input}, args.format)
        if args.output:
            with open(args.output, "wb") as f:
                f.write(data)
        else:
            sys.stdout.buffer.write(data)
        return 0
    if args.command == "serve":
        server = gateway.serve(args.host, args.port)
        print(f"serving on http://{args.host}:{server.server_address[1]}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return 0
    report = gateway.ci(args.inputs, args.format, args.out_dir, args.report)
    print(json.dumps({k: report[k] for k in ("format", "total", "failed")}))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(cli())
