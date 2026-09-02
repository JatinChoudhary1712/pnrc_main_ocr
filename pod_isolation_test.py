"""
Isolation Test 1 — run ON THE POD, hitting the backend directly (no browser, no proxy).

Mimics batch_upload.html's real pattern: submit PDFs in sequential batches of BATCH
to POST /process, then poll GET /process/{job_id} until every job is done/error.
Reports per-batch HTTP status + wall time, per-PDF submit/finish timing, and a summary.

    export API_KEY=<the pod's API_KEY>
    python3 pod_isolation_test.py test1                 # a folder of .pdf
    python3 pod_isolation_test.py test1/*.pdf           # an explicit list
    BATCH=1 python3 pod_isolation_test.py test1         # one file per request
    BASE_URL=http://127.0.0.1:8080 BATCH=6 TIMEOUT=3600 python3 pod_isolation_test.py test1

Exit code 0 only if: 0 submit failures AND 0 job-status=error AND 0 never-finished.
"""
import glob
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY") or sys.exit("set API_KEY (the pod's env value)")
BATCH = int(os.environ.get("BATCH", "6"))
POLL = float(os.environ.get("POLL", "2"))
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))  # seconds to wait for all jobs

# ---- resolve inputs -------------------------------------------------------
paths = []
for a in sys.argv[1:]:
    if os.path.isdir(a):
        paths += glob.glob(os.path.join(a, "*.pdf")) + glob.glob(os.path.join(a, "*.PDF"))
    else:
        paths += glob.glob(a)
paths = sorted(p for p in dict.fromkeys(paths) if p.lower().endswith(".pdf"))
if not paths:
    sys.exit("no PDFs matched")

headers = {"X-API-Key": API_KEY}
print(f"{len(paths)} PDF(s)  ->  {BASE_URL}/process   batch={BATCH}\n")

jobs = {}            # job_id -> dict
submit_fail = []     # pdf names that never got a job_id
t0 = time.time()

with httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0)) as c:
    for bi in range(0, len(paths), BATCH):
        chunk = paths[bi:bi + BATCH]
        blobs = [(os.path.basename(p), open(p, "rb").read()) for p in chunk]
        mb = sum(len(b) for _, b in blobs) / 1e6
        files = [("files", (name, data, "application/pdf")) for name, data in blobs]
        n = bi // BATCH + 1
        b0 = time.time()
        try:
            r = c.post(f"{BASE_URL}/process", headers=headers, files=files)
            dt = time.time() - b0
            print(f"batch {n:>3}: {len(chunk)} files {mb:6.2f} MB  ->  HTTP {r.status_code}  in {dt:6.2f}s")
            if r.status_code != 202:
                print(f"          body: {r.text[:400]}")
                submit_fail += [name for name, _ in blobs]
                continue
            for j in r.json().get("jobs", []):
                if j.get("job_id"):
                    jobs[j["job_id"]] = {
                        "name": j["pdf_name"], "submit_s": round(time.time() - t0, 1),
                        "done_s": None, "status": None, "pages": None, "errs": None, "proc_s": None,
                    }
                else:
                    submit_fail.append(j.get("pdf_name"))
                    print(f"          rejected {j.get('pdf_name')}: {j.get('error')}")
        except Exception as e:
            dt = time.time() - b0
            print(f"batch {n:>3}: EXCEPTION after {dt:6.2f}s  ->  {e!r}")
            submit_fail += [name for name, _ in blobs]

    print(f"\n{len(jobs)} job(s) queued, {len(submit_fail)} submit failure(s). polling every {POLL}s...\n")

    deadline = time.time() + TIMEOUT
    pending = set(jobs)
    while pending and time.time() < deadline:
        for jid in list(pending):
            try:
                s = c.get(f"{BASE_URL}/process/{jid}", headers=headers).json()
            except Exception:
                continue
            st = s.get("status")
            if st not in ("done", "error"):
                continue
            m = (s.get("result") or {}).get("metrics") or {}
            j = jobs[jid]
            j.update(done_s=round(time.time() - t0, 1), status=st,
                     pages=m.get("sheet_count"), errs=m.get("error_page_count"),
                     proc_s=m.get("processing_seconds"))
            pending.discard(jid)
            ok = st == "done" and not (j["errs"] or 0)
            print(f"  {'OK  ' if ok else 'FAIL'} {j['name']:<28} {st:<5} "
                  f"submit@{j['submit_s']}s finish@{j['done_s']}s proc={j['proc_s']}s "
                  f"pages={j['pages']} errpages={j['errs']}"
                  + ("" if st != "error" else f"  ERR: {s.get('error')}"))
        if pending:
            time.sleep(POLL)

# ---- summary ------------------------------------------------------------
done_clean = sum(1 for j in jobs.values() if j["status"] == "done" and not (j["errs"] or 0))
done_errpg = sum(1 for j in jobs.values() if j["status"] == "done" and (j["errs"] or 0))
job_error = sum(1 for j in jobs.values() if j["status"] == "error")
never = [j["name"] for j in jobs.values() if j["status"] is None]

print("\n==================== SUMMARY ====================")
print(f"PDFs given ............. {len(paths)}")
print(f"submit failures ....... {len(submit_fail)}  {submit_fail or ''}")
print(f"jobs queued ........... {len(jobs)}")
print(f"done, no error pages .. {done_clean}")
print(f"done, WITH error pages  {done_errpg}")
print(f"job status = error .... {job_error}")
print(f"never finished ........ {len(never)}  {never or ''}")
print(f"wall time ............. {round(time.time() - t0, 1)}s")
print("================================================")

sys.exit(0 if not submit_fail and not never and not job_error else 1)
