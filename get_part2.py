import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent))
from nebius_client import NebiusNvidiaClient

client = NebiusNvidiaClient()

system_prompt = """You are a Senior Principal AI Systems & GPU Compiler Architect at NVIDIA and elite judge for the Nebius x NVIDIA Global AI Hackathon ($50,000 in prizes). Respond in clear, articulate Spanish."""

prompt = """
Proponnos tus 3 MEJORES IDEAS GANADORAS alternativas o complementarias para el Nebius x NVIDIA Global AI Hackathon (compitiendo por el 1.er puesto de $20,000 en efectivo o los tracks con Jetson Orin Nano).

Para cada una de las 3 ideas, detalla:
1. Nombre y Track oficial (Coding & Agentic Engineering, Best Apps & Agents, Personal AI).
2. Tesis del problema real y solución técnica.
3. Arquitectura exacta: cómo utiliza NVIDIA Nemotron y la infraestructura de Nebius Token Factory.
4. Tu auditoría como juez: Puntuación del 1 al 10 y el por qué ganaría frente a los cientos de proyectos mediocres del hackathon.
5. El factor "efecto WOW" para la demo del jurado.
"""

res = client.complete(
    prompt=prompt,
    system_prompt=system_prompt,
    model="nvidia/Nemotron-3_5-Lightning",
    temperature=0.6,
    max_tokens=3000
)

print(res["content"])
