import os

DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")
CORPUS_GEN_MODEL = os.environ.get("CORPUS_GEN_MODEL", "claude-sonnet-5")
MAX_CHUNK_TOKENS = 500

HAIKU_INPUT_COST_PER_MTOK = 1.00
HAIKU_OUTPUT_COST_PER_MTOK = 5.00
