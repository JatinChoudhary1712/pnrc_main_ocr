import concurrent.futures
import json
import os
import sys
import time
import threading
from pathlib import Path

import httpx
import webview


def load_dotenv(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


exe_dir = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
load_dotenv(exe_dir / ".env")

DEFAULT_API_BASE = os.environ.get("PNRC_API_BASE", "https://jatin-veritos--pnrc-main-ocr-fastapi-app.modal.run")
MAX_IN_FLIGHT = 10  # Modal's hard container cap; keep <= that so we never over-submit
POLL_INTERVAL = 1.2
RESULT_MAX_ATTEMPTS = 60
RETRY_DELAY = 0.6
FRIENDLY_ERROR = "Unable to process this PDF. Please verify the PDF and try again."
SYSTEM_FILES = {".ds_store", "thumbs.db", "desktop.ini"}

ADMIN_USERNAME = "veritosadmin"
ADMIN_PASSWORD = "adminveritos"
USER_USERNAME = "userveritos"
USER_PASSWORD = "userveritos"

CONFIG_DIR = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "PNRC-OCR"
CONFIG_PATH = CONFIG_DIR / "config.json"


def load_config():
    if not CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(data):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_hidden(name: str) -> bool:
    return name.startswith(".") or name.lower() in SYSTEM_FILES


def fetch_json_with_retry(client, method, url, build_kwargs):
    """Send a request and parse JSON, retrying once on network failure or a
    non-JSON body (e.g. Modal's edge returning a plain-text routing error).
    Returns (status_code, body) on success, or (None, None) if both attempts fail.
    """

    def attempt():
        resp = client.request(method, url, **build_kwargs())
        return resp.status_code, resp.json()

    try:
        return attempt()
    except Exception:
        time.sleep(RETRY_DELAY)
        try:
            return attempt()
        except Exception:
            return None, None


class Api:
    def __init__(self):
        self.window = None
        self.cancelled = False
        config = load_config()
        self.api_base = config.get("api_base") or DEFAULT_API_BASE
        self.api_key = config.get("api_key", "")
        self.gpu_type = config.get("gpu_type", "L4")
        self.max_containers = config.get("max_containers", 6)
        self.scaledown_window = config.get("scaledown_window", 15)

    def login(self, username, password):
        return {"ok": username == USER_USERNAME and password == USER_PASSWORD}

    def admin_login(self, username, password):
        return {"ok": username == ADMIN_USERNAME and password == ADMIN_PASSWORD}

    def get_admin_settings(self):
        return {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "gpu_type": self.gpu_type,
            "max_containers": self.max_containers,
            "scaledown_window": self.scaledown_window,
        }

    def save_admin_settings(self, settings):
        self.api_base = settings.get("api_base") or self.api_base
        self.api_key = settings.get("api_key", self.api_key)
        self.gpu_type = settings.get("gpu_type", self.gpu_type)
        self.max_containers = settings.get("max_containers", self.max_containers)
        self.scaledown_window = settings.get("scaledown_window", self.scaledown_window)
        save_config({
            "api_base": self.api_base,
            "api_key": self.api_key,
            "gpu_type": self.gpu_type,
            "max_containers": self.max_containers,
            "scaledown_window": self.scaledown_window,
        })
        return {"ok": True}

    def pick_input_folder(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return {"ok": False, "cancelled": True}
        folder = result[0]

        entries = [e for e in os.listdir(folder) if not is_hidden(e)]
        subfolders = [e for e in entries if os.path.isdir(os.path.join(folder, e))]
        if subfolders:
            return {
                "ok": False,
                "error": f"Subfolders found — only flat PDF files are allowed: {', '.join(subfolders[:3])}",
            }

        non_pdf = [e for e in entries if not e.lower().endswith(".pdf")]
        if non_pdf:
            return {
                "ok": False,
                "error": f"{len(non_pdf)} non-PDF file(s) found: {', '.join(non_pdf[:3])}",
            }

        pdfs = sorted(entries)
        if not pdfs:
            return {"ok": False, "error": "No PDF files found in this folder."}

        return {"ok": True, "folder": folder, "count": len(pdfs)}

    def pick_output_folder(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return {"ok": False, "cancelled": True}
        return {"ok": True, "folder": result[0]}

    def start_processing(self, input_folder, output_folder):
        if not self.api_key:
            return {"ok": False, "error": "No API key configured. Ask your administrator to set one in the admin panel."}
        threading.Thread(
            target=self._run_batch,
            args=(input_folder, output_folder),
            daemon=True,
        ).start()
        return {"ok": True}

    def cancel(self):
        self.cancelled = True
        return {"ok": True}

    def _emit(self, event, payload):
        self.window.evaluate_js(
            f"window.onDesktopEvent({json.dumps(event)}, {json.dumps(payload)})"
        )

    def _run_batch(self, input_folder, output_folder):
        self.cancelled = False
        pdf_names = sorted(
            f for f in os.listdir(input_folder)
            if f.lower().endswith(".pdf") and not is_hidden(f)
        )
        total = len(pdf_names)
        self._emit("start", {"total": total})

        headers = {"X-API-Key": self.api_key}
        state = {"done": 0, "ok": 0, "err": 0}
        state_lock = threading.Lock()

        def process_one(client, pdf_name):
            if self.cancelled:
                return

            try:
                jobs = self._submit_batch(client, input_folder, [pdf_name], headers)
                job = jobs[0]
                if job.get("status") == "error" and "job_id" not in job:
                    result = {"pdf_name": pdf_name, "status": "error", "error": job.get("error", "Failed to queue.")}
                else:
                    result = self._track_job(client, job, headers)
            except Exception:
                result = {"pdf_name": pdf_name, "status": "error", "error": FRIENDLY_ERROR}

            out_path = Path(output_folder) / f"{Path(pdf_name).stem}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            is_err = result.get("status") == "error"
            with state_lock:
                state["done"] += 1
                state["ok"] += 0 if is_err else 1
                state["err"] += 1 if is_err else 0
                snapshot = dict(state)

            self._emit("file_done", {
                "pdf_name": pdf_name,
                "status": "error" if is_err else "done",
                "done": snapshot["done"],
                "total": total,
                "ok": snapshot["ok"],
                "err": snapshot["err"],
                "output_path": str(out_path),
            })

        # Sliding window: up to MAX_IN_FLIGHT PDFs in flight at once. The pool
        # pulls the next queued PDF the instant a worker frees up, instead of
        # waiting for a whole batch to finish.
        with httpx.Client(timeout=60) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as executor:
                futures = [executor.submit(process_one, client, name) for name in pdf_names]
                concurrent.futures.wait(futures)

        self._emit("complete", {
            "done": state["done"], "total": total,
            "ok": state["ok"], "err": state["err"], "cancelled": self.cancelled,
        })

    def _submit_batch(self, client, input_folder, chunk, headers):
        file_bytes = {n: Path(input_folder, n).read_bytes() for n in chunk}

        def build_kwargs():
            return {
                "headers": headers,
                "files": [("files", (n, file_bytes[n], "application/pdf")) for n in chunk],
            }

        status, body = fetch_json_with_retry(client, "POST", f"{self.api_base}/process/batch", build_kwargs)

        if body is None:
            return [{"pdf_name": n, "status": "error", "error": FRIENDLY_ERROR} for n in chunk]
        if status >= 400:
            detail = body.get("detail", f"HTTP {status}") if isinstance(body, dict) else f"HTTP {status}"
            return [{"pdf_name": n, "status": "error", "error": str(detail)} for n in chunk]

        return body.get("jobs", [])

    def _track_job(self, client, job, headers):
        job_id = job["job_id"]
        call_id = job["call_id"]
        pdf_name = job["pdf_name"]

        while True:
            if self.cancelled:
                return {"pdf_name": pdf_name, "status": "error", "error": "Cancelled by user."}

            status, body = fetch_json_with_retry(
                client, "GET", f"{self.api_base}/process/progress/{job_id}",
                lambda: {"headers": headers},
            )
            if body is None:
                return {"pdf_name": pdf_name, "status": "error", "error": FRIENDLY_ERROR}

            progress = {"stage": "queued"} if status == 404 else body

            if progress.get("stage") == "error":
                return {"pdf_name": pdf_name, "status": "error", "error": progress.get("error", "Unknown error")}
            if progress.get("stage") == "done":
                break
            time.sleep(POLL_INTERVAL)

        for _ in range(RESULT_MAX_ATTEMPTS):
            if self.cancelled:
                return {"pdf_name": pdf_name, "status": "error", "error": "Cancelled by user."}

            status, body = fetch_json_with_retry(
                client, "GET", f"{self.api_base}/process/result/{call_id}",
                lambda: {"headers": headers},
            )
            if body is None:
                return {"pdf_name": pdf_name, "status": "error", "error": FRIENDLY_ERROR}
            if status == 202:
                time.sleep(POLL_INTERVAL)
                continue
            if status >= 400:
                detail = body.get("detail", f"HTTP {status}") if isinstance(body, dict) else f"HTTP {status}"
                return {"pdf_name": pdf_name, "status": "error", "error": str(detail)}
            return {"pdf_name": pdf_name, **body}

        return {"pdf_name": pdf_name, "status": "error", "error": "Timed out waiting for result after processing finished."}


def main():
    api = Api()
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    html_path = base_dir / "ui.html"
    window = webview.create_window("PNRC OCR", str(html_path), js_api=api, width=760, height=780, min_size=(600, 600))
    api.window = window
    webview.start(gui="edgechromium")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import tempfile
        import traceback
        Path(tempfile.gettempdir(), "PNRC-OCR-crash.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
