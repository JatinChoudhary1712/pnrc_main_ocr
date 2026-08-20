import ollama

from src.config import OLLAMA_HOST

ollama_client = ollama.Client(host=OLLAMA_HOST)
