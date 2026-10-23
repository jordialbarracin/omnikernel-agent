import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from nebius_client import NebiusNvidiaClient

client = NebiusNvidiaClient()

system_prompt = "You are a Senior Principal AI Systems & GPU Compiler Architect at NVIDIA and hackathon judge. Be concise, direct, and write in Spanish."
prompt = """
Propón 3 ideas ganadoras para el Nebius x NVIDIA Global AI Hackathon ($50,000 en premios).
Para cada idea da:
1. Nombre y Track.
2. Problema y Solución.
3. Arquitectura técnica con Nebius y modelos NVIDIA (Nemotron / Triton).
4. Nota del 1 al 10 y por qué gana.
"""

res = client.complete(
    prompt=prompt,
    system_prompt=system_prompt,
    model="nvidia/Nemotron-3_5-Lightning",
    temperature=0.6,
    max_tokens=4096
)

with open(r"C:\Users\jordi\.gemini\antigravity\scratch\nebius-nvidia-agent\ideas_audited.txt", "w", encoding="utf-8") as f:
    f.write(res["content"])

print(f"Done! Characters: {len(res['content'])}, Tokens: {res['completion_tokens']}")
