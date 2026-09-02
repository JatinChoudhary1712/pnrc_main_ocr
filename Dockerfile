# RunPod template: vLLM OCR server (:8000, internal) + FastAPI backend (:8080).
# Muse-Glimmer-30B needs vLLM nightly (its muse_glimmer parsers aren't in stable yet).
ARG BASE_IMAGE=vllm/vllm-openai:nightly
FROM ${BASE_IMAGE}

WORKDIR /app

# Backend-only deps (torch/transformers/numpy/pillow come from the base image).
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart python-dotenv httpx pymupdf scikit-learn

# Bake DINOv2 so the pod never downloads it at runtime (CPU inference).
RUN python3 -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"

COPY src ./src
COPY knowledge_base ./knowledge_base
COPY start.sh ./start.sh
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# DINOv2 is baked above and loads from cache. The OCR model (30B) is NOT baked,
# so leave HF online — start.sh downloads it on first boot. Set HF_TOKEN in the
# pod env if meta-models/Muse-Glimmer-30B is gated.
ENV VLLM_BASE_URL=http://127.0.0.1:8000/v1

EXPOSE 8000 8080
ENTRYPOINT ["bash", "./start.sh"]
