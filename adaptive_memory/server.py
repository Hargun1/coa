"""Loopback-only demo server. The scientific runner is independent of this UI."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config
from .experiments import ROOT, summarize
from .policies import POLICIES
from .simulator import simulate
from .workloads import WORKLOADS, generate

RUN_LOCK = threading.Lock()


def run_demo(payload):
    allowed = {"workload", "requests", "warmup", "seed", "l3_lines", "horizon"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unsupported experiment fields")
    workload = payload.get("workload", "phase_shift")
    if workload not in WORKLOADS:
        raise ValueError("Choose a supported workload")
    defaults = {"requests": 15000, "warmup": 3000, "seed": 11, "l3_lines": 512, "horizon": 256}
    for key, default in defaults.items():
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        defaults[key] = value
    if not 100 <= defaults["requests"] <= 50000:
        raise ValueError("requests must be between 100 and 50,000")
    if not 0 <= defaults["warmup"] < defaults["requests"]:
        raise ValueError("warmup must be nonnegative and smaller than requests")
    if not 0 <= defaults["seed"] <= 99999:
        raise ValueError("seed must be between 0 and 99,999")
    if defaults["l3_lines"] not in (256, 512, 1024):
        raise ValueError("L3 capacity must be 256, 512, or 1024 lines")
    if not 8 <= defaults["horizon"] <= 2048:
        raise ValueError("horizon must be between 8 and 2048 LLC requests")
    config = Config(l3_lines=defaults["l3_lines"], horizon=defaults["horizon"])
    addresses = generate(workload, defaults["requests"], defaults["seed"])
    runs = []
    for policy in POLICIES:
        result = simulate(addresses, config, policy, defaults["warmup"], defaults["seed"])
        result["workload"] = workload
        runs.append(result)
    return {"manifest": {"run_count": len(runs), "trace_type": "Interactive single-seed synthetic demo",
                         "study": {"seeds": [defaults["seed"]], "warmup": defaults["warmup"],
                                   "measured_requests": defaults["requests"] - defaults["warmup"]}},
            "summary": summarize(runs), "runs": runs}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, content, content_type="application/json; charset=utf-8"):
        encoded = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.respond(204, b"", "image/x-icon")
            return
        routes = {"/": (ROOT / "web/index.html", "text/html; charset=utf-8"),
                  "/app.js": (ROOT / "web/app.js", "text/javascript; charset=utf-8"),
                  "/style.css": (ROOT / "web/style.css", "text/css; charset=utf-8"),
                  "/api/results": (ROOT / "results/benchmark.json", "application/json; charset=utf-8"),
                  "/paper.pdf": (ROOT / "output/pdf/research_paper.pdf", "application/pdf"),
                  "/health": (None, None)}
        if self.path == "/health":
            self.respond(200, {"status": "ok", "application": "Adaptive Memory Lab"})
            return
        route = routes.get(self.path)
        if not route:
            self.respond(404, {"error": "Not found"})
        elif not route[0].is_file():
            self.respond(404, {"error": "Artifact not generated yet; run benchmark and report"})
        else:
            self.respond(200, route[0].read_bytes(), route[1])

    def do_POST(self):
        if self.path != "/api/run":
            self.respond(404, {"error": "Not found"})
            return
        if self.headers.get("X-Lab-Request") != "1" or self.headers.get_content_type() != "application/json":
            self.respond(403, {"error": "Use the local dashboard to run experiments"})
            return
        if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            self.respond(403, {"error": "Origin mismatch"})
            return
        if not RUN_LOCK.acquire(blocking=False):
            self.respond(429, {"error": "An experiment is already running"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request body must be between 1 and 4096 bytes")
            payload = json.loads(self.rfile.read(length))
            self.respond(200, run_demo(payload))
        except (ValueError, TypeError, KeyError) as error:
            self.respond(400, {"error": str(error)})
        finally:
            RUN_LOCK.release()


def serve(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Adaptive Memory Lab: http://127.0.0.1:{port}", flush=True)
    print("Press Ctrl+C to stop. All simulations run locally.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

