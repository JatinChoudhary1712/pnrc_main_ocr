# RunPod deployment

One image, two processes: vLLM OCR on `:8000` (pod-internal) and the FastAPI
backend on `:8080` (exposed). `start.sh` launches vLLM, waits for its `/health`,
then starts uvicorn. DINOv2 runs on CPU and is baked into the image.

## 1. Build & push

```bash
docker build -t <registry>/pnrc-ocr:latest .
docker push <registry>/pnrc-ocr:latest
```

Base image is `vllm/vllm-openai:nightly` (Muse-Glimmer's `muse_glimmer` parsers
are not in the stable tag yet). Pin a dated nightly for reproducibility:
`--build-arg BASE_IMAGE=vllm/vllm-openai:nightly-<sha>`

Local smoke test (needs nvidia-container-toolkit):
```bash
docker run --gpus all -p 8080:8080 \
  -e MODEL_ID=<repo-or-path> -e API_KEY=test <registry>/pnrc-ocr:latest
```

## 2. Create the RunPod template

- **Container Image:** `<registry>/pnrc-ocr:latest`
- **Start command:** leave empty (the image's ENTRYPOINT runs `start.sh`)
- **Container Disk:** 150 GB (running image + ~60 GB of Muse-Glimmer-30B weights)
- **Expose HTTP Ports:** `8080`
- **Environment variables:**

| Var | Required | Example |
|---|---|---|
| `MODEL_ID` | yes | `meta-models/Muse-Glimmer-30B` (HF repo or `/path`) |
| `API_KEY` | yes | shared secret for the `X-API-Key` header |
| `HF_TOKEN` | if gated | HF access token for the model repo |
| `SERVED_MODEL_NAME` | no | `muse-glimmer` (default) |
| `VLLM_EXTRA_ARGS` | recommended | `--tensor-parallel-size <#GPUs> --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 64` |
| `OCR_CONCURRENCY` | no | `12` |
| `OCR_PAGE_RETRIES` | no | `2` |
| `MAX_FILES_PER_BATCH` | no | `8` |
| `OUTPUT_DIR` | no | `/workspace/output` (mount a volume there to keep the JSON) |

## 3. Launch a pod

Muse-Glimmer-30B in bf16 is ~60 GB of weights plus KV cache. Options:

- **1× RTX PRO 6000 (96 GB):** works with `--max-model-len 32768` (or lower).
  Leave `--tensor-parallel-size` at 1.
- **2–4× GPUs:** set `--tensor-parallel-size` to the GPU count in
  `VLLM_EXTRA_ARGS` and you can raise `--max-model-len`.
- **1× RTX 5090 (32 GB):** only the NVFP4 quantized variant fits — point
  `MODEL_ID` at that repo instead.

First boot downloads the weights and loads them (10+ min) — `start.sh` blocks
the backend until vLLM's `/health` passes, and RunPod restarts the pod if vLLM
dies during startup (e.g. OOM — lower `--max-model-len`).

## 4. Use it

RunPod gives you `https://<pod-id>-8080.proxy.runpod.net`.

**Preferred: one PDF per request (`/upload`).** Each request is small, so a slow
uplink can't push it past the proxy's request-duration limit, and a client
retry carrying the same `Idempotency-Key` is de-duplicated instead of starting a
second OCR job.

```bash
curl -H "X-API-Key: $API_KEY" -H "Idempotency-Key: $(sha256sum doc.pdf | cut -c1-64)" \
  -F "file=@doc.pdf" https://<pod-id>-8080.proxy.runpod.net/upload
# -> {"job_id":"...","filename":"doc.pdf","status":"queued","idempotency_key":"..."}

curl -H "X-API-Key: $API_KEY" \
  https://<pod-id>-8080.proxy.runpod.net/process/<job_id>
```

`POST /process` (multi-file, whole batch in one multipart request) still exists
for `smoke_test.py` / `backend_test.html`, but it is the fragile path on a slow
link — prefer `/upload`.

`GET /healthz` (no key) and `GET /metrics` (key) expose queue depth, in-flight
OCR, and job counts.

Open `batch_upload.html` (served over `http://localhost`, not `file://`) for the
folder-batch UI — it uploads one PDF at a time with configurable concurrency.

### Debugging the proxy

To rule the RunPod HTTPS proxy in or out, add `8080` under the template's
**"Expose TCP Ports"**, restart, and hit the mapping shown in the pod's Connect
panel directly: `curl ... http://<pod-ip>:<tcp-port>/upload`. The app binds
`0.0.0.0:8080` already, so no code change is needed. Don't make production depend
on this — it's for isolation only.

## Notes

- `.env` is not used on the pod — RunPod injects the vars directly. `config.py`
  calls `load_dotenv` which silently no-ops when the file is absent.
- `OUTPUT_DIR` on the container disk is lost when the pod stops; mount a network
  volume there to keep completed-job JSON.
- Code change = rebuild + push + restart the pod (the image is immutable).
- vLLM is bound to `127.0.0.1`, so only `8080` is reachable from outside.
