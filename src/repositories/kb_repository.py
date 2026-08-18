import json
import numpy as np

from src.config import EMBEDDINGS_FILE, METADATA_FILE


def save_kb(embeddings, metadata):
    """Save embeddings and metadata to disk."""

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def load_kb():
    """Load embeddings and metadata from disk."""

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    return embeddings, metadata

def kb_exists():
    return EMBEDDINGS_FILE.exists() and METADATA_FILE.exists()

def kb_image_count():
    if not kb_exists():
        return 0
    with open(METADATA_FILE) as f:
        return len(json.load(f))
