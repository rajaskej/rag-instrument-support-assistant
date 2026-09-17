import os

DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-flash-lite-latest")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-flash-lite-latest")
CORPUS_GEN_MODEL = os.environ.get("CORPUS_GEN_MODEL", "gemini-flash-lite-latest")
MAX_CHUNK_TOKENS = 500
