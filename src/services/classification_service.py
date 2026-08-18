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


def classify_pdf(pdf_bytes):
    embeddings, metadata = load_kb()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    results = []
    page_images = {}

    for page_number in range(len(doc)):
        pix = doc[page_number].get_pixmap(dpi=PDF_DPI)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        result = classify_image(image, embeddings, metadata)

        page_images[page_number + 1] = pix

        results.append(
            {
                "page": page_number + 1,
                **result,
            }
        )

    doc.close()

    return results, page_images
