import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load .env from current directory or nearest parent containing .env
_env_path = find_dotenv()
if _env_path:
    load_dotenv(_env_path)
else:
    # Fallback: try project root (two levels up from this file)
    project_root_env = Path(__file__).resolve().parents[1] / '.env'
    if project_root_env.exists():
        load_dotenv(project_root_env)

def _get_env_str(*names: str):
    for name in names:
        val = os.getenv(name)
        if isinstance(val, str) and val.strip():
            return val.strip().strip('"').strip("'")
    return None

class Settings:
    # API Keys
    OPENAI_API_KEY = _get_env_str("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_API_TOKEN")
    
    # Paths
    MIMIC_SOURCE = os.getenv("MIMIC_SOURCE", "csv").lower()  # csv | sqlite
    MIMIC_DB_PATH = os.getenv("MIMIC_DB_PATH", "./data/mimic/mimic.db")
    MIMIC_CSV_DIR = os.getenv("MIMIC_CSV_DIR", "./data/mimic")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    
    # Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    LOCAL_ONLY = os.getenv("LOCAL_ONLY", "false").lower() == "true"
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "500"))
    
    # System settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PHI_DETECTION = os.getenv("ENABLE_PHI_DETECTION", "true").lower() == "true"
    AUDIT_LOGGING = os.getenv("AUDIT_LOGGING", "true").lower() == "true"

settings = Settings()