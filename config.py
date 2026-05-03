import os
from dotenv import load_dotenv

# Load .env from the project root (not CWD)
_project_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_project_root, ".env"), override=True)

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_groq_api_key() -> str:
    """Get the Groq API key dynamically (supports sidebar override)."""
    return os.getenv("GROQ_API_KEY", "")

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
