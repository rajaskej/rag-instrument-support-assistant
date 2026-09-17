import os

DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-2.5-flash-lite")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash-lite")
CORPUS_GEN_MODEL = os.environ.get("CORPUS_GEN_MODEL", "gemini-2.5-flash-lite")
MAX_CHUNK_TOKENS = 500
