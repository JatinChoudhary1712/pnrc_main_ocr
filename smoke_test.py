"""Smoke test: upload PDFs to a running pod, poll every job, assert OCR produced text.

    python smoke_test.py                      # default pod + 1.pdf..5.pdf
    python smoke_test.py a.pdf b.pdf          # specific files
    BASE_URL=https://... API_KEY=... python smoke_test.py
"""
import os
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252

BASE_URL = os.environ.get("BASE_URL", "https://pl3ygp2tnh59el-8080.proxy.runpod.net").rstrip("/")
API_KEY = os.environ.get("API_KEY", "veritos-2121321")
TIMEOUT = int(os.environ.get("TIMEOUT", "1200"))  # seconds to wait for all jobs
PDFS = sys.argv[1:] or ["1.pdf", "2.pdf", "3.pdf", "4.pdf", "5.pdf"]


def main():
    headers = {"X-API-Key": API_KEY}
    files = [("files", (os.path.basename(p), open(p, "rb"), "application/pdf")) for p in PDFS]
    r = httpx.post(f"{BASE_URL}/process", headers=headers, files=files, timeout=300)
    assert r.status_code == 202, f"submit failed: {r.status_code} {r.text}"

    jobs = {j["job_id"]: j["pdf_name"] for j in r.json()["jobs"] if "job_id" in j}
    print(f"queued {len(jobs)} job(s): {', '.join(jobs.values())}\n")

    results = {}
    deadline = time.time() + TIMEOUT
    while len(results) < len(jobs):
        for jid, name in jobs.items():
            if jid in results:
                continue
            s = httpx.get(f"{BASE_URL}/process/{jid}", headers=headers, timeout=30).json()
            if s["status"] in ("done", "error"):
                results[jid] = s
                print(f"  {name}: {s['status']}")
            else:
                print(f"  {name}: {s['stage']} {s['pages_done']}/{s['pages_total']}")
        if len(results) < len(jobs):
            if time.time() > deadline:
                sys.exit(f"timed out after {TIMEOUT}s")
            time.sleep(5)

    print()
    failed = 0
    for jid, name in jobs.items():
        s = results[jid]
        if s["status"] == "error":
            print(f"FAIL {name}: {s.get('error')}")
            failed += 1
            continue
        res = s["result"]
        pages = res["pages"]
        with_text = sum(1 for p in pages if p["status"] == "filled" and p["text"])
        errs = sum(1 for p in pages if p["status"] == "error")
        m = res["metrics"]
        print(f"{'OK  ' if not errs else 'FAIL'} {name}: {len(pages)} pages, "
              f"{with_text} with text, {errs} errors, {m['processing_seconds']}s")
        if errs or not with_text:
            failed += 1

    print()
    if failed:
        sys.exit(f"{failed}/{len(jobs)} file(s) failed")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
