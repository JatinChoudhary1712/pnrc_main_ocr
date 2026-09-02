from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MARGIN, PDF_DPI, TOP_N
from src.repositories.kb_repository import load_kb
from src.services.embedding_service import get_embedding


def classify_image(image, embeddings=None, metadata=None):
    if embeddings is None or metadata is None:
        embeddings, metadata = load_kb()

    labels = np.array([m["label"] for m in metadata])

    empty_embeddings = embeddings[labels == "empty"]
    filled_embeddings = embeddings[labels == "filled"]

    query_embedding = get_embedding(image).reshape(1, -1)

    empty_scores = cosine_similarity(query_embedding, empty_embeddings)[0]
    filled_scores = cosine_similarity(query_embedding, filled_embeddings)[0]

    empty_score = np.mean(np.sort(empty_scores)[::-1][:TOP_N])
    filled_score = np.mean(np.sort(filled_scores)[::-1][:TOP_N])

    is_filled = empty_score < filled_score + MARGIN

    return {
        "prediction": "filled" if is_filled else "empty",
        "empty_score": float(empty_score),
        "filled_score": float(filled_score),
    }


def iter_classified_pages(pdf):
    """Yield (page_number, result_dict, pixmap) one page at a time so a caller
    can act on each page the moment it's classified.

    `pdf` is either raw PDF bytes or a filesystem path (str/Path) to a PDF —
    passing a path lets PyMuPDF read the file directly instead of holding the
    whole document in memory.
    """
    embeddings, metadata = load_kb()
    if isinstance(pdf, (str, Path)):
        doc = fitz.open(pdf)
    else:
        doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        for page_number in range(len(doc)):
            pix = doc[page_number].get_pixmap(dpi=PDF_DPI)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield page_number + 1, classify_image(image, embeddings, metadata), pix
    finally:
        doc.close()


def classify_pdf(pdf_bytes):
    results = []
    page_images = {}
    for page_number, result, pix in iter_classified_pages(pdf_bytes):
        page_images[page_number] = pix
        results.append({"page": page_number, **result})
    return results, page_images
