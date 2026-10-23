import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent))
from nebius_client import NebiusNvidiaClient

client = NebiusNvidiaClient()

system_prompt = "You are a Senior Principal AI Systems & GPU Compiler Architect at NVIDIA and hackathon judge. Be concise, direct, and write in Spanish."
prompt = """
Para el proyecto 'OmniKernel' en el Nebius x NVIDIA Hackathon:
¿Cómo diseñar la demo perfecta paso a paso para que los jueces queden alucinados en 2 minutos de video/presentación? Da el guion exacto de 2 minutos y qué mostrar en pantalla en cada segundo.
"""

res = client.complete(
    prompt=prompt,
    system_prompt=system_prompt,
    model="nvidia/Nemotron-3_5-Lightning",
    temperature=0.5,
    max_tokens=1500
)

with open(r"C:\Users\jordi\.gemini\antigravity\scratch\nebius-nvidia-agent\omnikernel_demo_script.txt", "w", encoding="utf-8") as f:
    f.write(res["content"])

print("Demo script saved successfully!")
