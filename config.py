import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "nvidia/Llama-3.1-Nemotron-70B-Instruct")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

def validate_config() -> bool:
    """Check if the API key is configured."""
    return bool(NEBIUS_API_KEY and not NEBIUS_API_KEY.startswith("tu_"))
