# src/utils/text_utils.py
import re


def clean_repetitions(text: str) -> str:
    text = re.sub(r"\b(\S+)(?:\s+\1\b){2,}", r"\1", text)
    lines = text.split("\n")

    deduped = []
    for line in lines:
        if deduped and line.strip() and line.strip() == deduped[-1].strip():
            continue
        deduped.append(line)
    text = "\n".join(deduped)

    paragraphs = text.split("\n\n")
    deduped = []
    for para in paragraphs:
        if deduped and para.strip() and para.strip() == deduped[-1].strip():
            continue
        deduped.append(para)

    return "\n\n".join(deduped)
