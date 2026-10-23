import time
from typing import List, Dict, Any, Generator, Optional
from openai import OpenAI
from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS, REQUEST_TIMEOUT, validate_config

class NebiusNvidiaClient:
    """
    Client for Nebius Token Factory inference serving NVIDIA open-source models.
    Compatible with OpenAI standard API format.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or NEBIUS_API_KEY
        self.base_url = base_url or NEBIUS_BASE_URL
        
        if not self.api_key:
            raise ValueError(
                "NEBIUS_API_KEY is not set. Please set it in your .env file or pass it to NebiusNvidiaClient."
            )
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT
        )
        
    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful and intelligent AI assistant.",
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Execute a standard completion and measure latency and throughput.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        latency = time.perf_counter() - start_time
        
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        return {
            "content": content,
            "latency_seconds": round(latency, 3),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "model": response.model
        }

    def stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful and intelligent AI assistant.",
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """
        Stream completion chunks in real time.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response_stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def list_models(self) -> List[str]:
        """
        Fetch available models on Nebius Token Factory.
        """
        try:
            models_data = self.client.models.list()
            return [m.id for m in models_data.data]
        except Exception as e:
            return [f"Error listing models: {e}"]
