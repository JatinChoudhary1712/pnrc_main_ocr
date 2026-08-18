from transformers import AutoImageProcessor, AutoModel
import torch

from src.config import MODEL_NAME, get_device

DEVICE = get_device()

print(f"Loading {MODEL_NAME} on {DEVICE}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully...")

def get_embedding(image):
    """Generate an embedding for a PIL image."""

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()[0]
