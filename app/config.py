import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import logging
import warnings

# Some libraries (e.g. FlashRank) switch on INFO logging for everything, flooding the terminal
for noisy_logger in ("httpx", "httpx2", "httpcore", "openai", "flashrank"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Harmless warning emitted by structured output when LangSmith tracing is on
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
MAX_REVISIONS = 2  # the writer can rewrite at most this many times
APPROVAL_SCORE = 8  # the critic's score needed to approve a draft
MAX_RESEARCH_ROUNDS = 2  # the researcher can run at most this many times

JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL_NAME)  # model that grades the answers


def get_judge_llm():
    return ChatOpenAI(model=JUDGE_MODEL, max_retries=3, timeout=120)

def get_llm():
    """Every agent gets its LLM from here, so changing models is a one-line edit."""
    return ChatOpenAI(
        model=MODEL_NAME,
        max_retries=3,   # retry rate-limit and network errors automatically
        timeout=90,      # give up on a single call after 90 seconds
    )

# --- RAG settings ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared by neighbouring chunks

MIN_CHUNK_CHARS = 80   # skip tiny chunks such as title pages or page numbers alone

RAG_CANDIDATES = 20      # results per individual search (vector or keyword)
RERANK_POOL = 30         # how many fused candidates the reranker reads
RAG_TOP_K = 5            # chunks finally returned
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"   # ~34 MB, runs on CPU
MIN_RELEVANCE = 0.01     # drop reranked chunks scoring below this

ORIGINAL_QUERY_WEIGHT = 3.0   # weight of the user's own question vs. each rewritten query in RRF
QUERY_EXPANSION = False   # alternative queries didn't improve retrieval in evaluation
RERANK_MODE = "filter"    # "filter": reranker only removes irrelevant chunks (hybrid order kept)
                          # "rank":   reranker also reorders them (scored worse in evaluation)


DATABASE_URL = os.getenv("DATABASE_URL", "")  # set: Postgres (Neon). Empty: local files
USE_POSTGRES = bool(DATABASE_URL)

EMBEDDING_DIM = 1536   # vector size of text-embedding-3-small (3072 for -large)

# --- Accounts (only used online, when DATABASE_URL is set) ---
ALLOWED_EMAILS = {e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()}
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()
LOGIN_REQUIRED = USE_POSTGRES  # local single-user mode needs no login