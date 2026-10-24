import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent))
from nebius_client import NebiusNvidiaClient

client = NebiusNvidiaClient()

system_prompt = """You are a Senior Principal AI Systems & GPU Compiler Architect at NVIDIA, acting as an elite judge for the Nebius x NVIDIA Global AI Hackathon ($50,000 cash and Jetson prizes).
Your evaluation must be rigorous, deeply technical, practical, and brutally honest. You value systems engineering, real GPU speedups, novel use of NVIDIA architectures (Triton, Nemotron, Tensor Cores, SRAM/HBM cache hierarchy), and production viability.
Respond in clear, articulate Spanish as requested by the lead builder."""

prompt = """
Act  a como juez t  cnico principal de NVIDIA en el Nebius x NVIDIA Global AI Hackathon.

Haz una AUDITOR  A EXHAUSTIVA DE 'OMNIKERNEL' (Puntaje del 1 al 10):
Concepto: Un agente aut  nomo de ingenier  a de compilaci  n y rendimiento GPU impulsado por NVIDIA Nemotron en Nebius Token Factory.

Flujo del sistema:
1. El usuario introduce un bloque de c  digo PyTorch est  ndar (ej. atenci  n dispersa, capas fusionadas LayerNorm+GELU+Linear, operadores MoE routing, o cuantizaci  n FP8/INT4 personalizada).
2. El agente (NVIDIA Nemotron en Nebius) analiza el AST de PyTorch, la coalescencia de memoria en HBM, el reuso de registros y la jerarqu  a de cach   SRAM/Shared Memory.
3. Sintetiza autom  ticamente un kernel optimizado en OpenAI Triton compilable para GPUs NVIDIA con Tensor Cores (ej. Hopper H100 / Ada Lovelace / Ampere).
4. Ejecuta un motor de verificaci  n matem  tica (equivalencia num  rica rigurosa con torch.allclose en FP16/BF16/FP8 y detecci  n de race conditions).
5. Produce un benchmark interactivo comparativo en tiempo real (Speedup 2x-4x, TFLOPS alcanzados y ahorro de VRAM).

Por favor audita esta idea con el m  ximo detalle t  cnico:
-   Qu   es exactamente un GPU Kernel en t  rminos sencillos pero rigurosos para que cualquier persona entienda su valor?
-   Cu  l es su nota del 1 al 10 ante el jurado de NVIDIA y Nebius, y por qu  ?
-   Cu  les son sus mayores fortalezas competitivas en el track 'Coding & Agentic Engineering'?
-   Cu  les son los riesgos t  cnicos reales y c  mo debemos mitigarlos para que el prototipo funcione impecable este fin de semana?
-   C  mo dise  ar la demo perfecta para que los jueces queden alucinados en 2 minutos?
"""

res = client.complete(
    prompt=prompt,
    system_prompt=system_prompt,
    model="nvidia/Nemotron-3_5-Lightning",
    temperature=0.5,
    max_tokens=3500
)

with open(r"C:\Users\jordi\.gemini\antigravity\scratch\nebius-nvidia-agent\omnikernel_audit.txt", "w", encoding="utf-8") as f:
    f.write(res["content"])

print(f"Audit completed in {res['latency_seconds']}s! Output saved to omnikernel_audit.txt ({res['completion_tokens']} tokens).")
