import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure we import config and client
sys.path.append(str(Path(__file__).parent))
from nebius_client import NebiusNvidiaClient

def main():
    print("=" * 70)
    print("  CONSULTANDO A NVIDIA NEMOTRON EN NEBIUS TOKEN FACTORY...")
    print("=" * 70)
    
    client = NebiusNvidiaClient()
    
    system_prompt = """You are a Senior Principal AI Systems & GPU Compiler Architect at NVIDIA, acting as an elite judge for the Nebius x NVIDIA Global AI Hackathon ($50,000 cash and Jetson prizes).
Your evaluation must be rigorous, deeply technical, practical, and brutally honest. You value systems engineering, real GPU speedups, novel use of NVIDIA architectures (Triton, Nemotron, CUDA/ROCm, Tensor Cores, HBM), and production viability.
Respond in clear, articulate Spanish as requested by the lead builder."""

    prompt = """
Actúa como juez técnico principal de NVIDIA en el Nebius x NVIDIA Global AI Hackathon.

1. AUDITORÍA EXHAUSTIVA DE 'OMNIKERNEL' (Puntaje del 1 al 10):
Concepto: Un agente autónomo de ingeniería de compilación y rendimiento GPU.
Flujo:
- El usuario introduce código PyTorch estándar (ej. atención dispersa, capas fusionadas, operadores MoE, cuantización personalizada).
- El agente (impulsado por Nemotron) analiza la coalescencia de memoria, jerarquía de caché SRAM/HBM y paralelismo de bloques.
- Sintetiza automáticamente un kernel en OpenAI Triton optimizado para GPUs NVIDIA (Tensor Cores / H100).
- Ejecuta una suite de verificación matemática (equivalencia numérica con tolerancia FP16/BF16/FP8 y comprobación de race conditions).
- Produce un benchmark interactivo comparativo (Speedup 2x-5x, TFLOPS alcanzados y ahorro de VRAM).

Por favor audita esta idea:
- ¿Cuál es su nota del 1 al 10 y por qué?
- ¿Cuáles son sus mayores fortalezas ante los jueces de NVIDIA y Nebius?
- ¿Cuáles son los riesgos técnicos reales y cómo debemos mitigarlos para que el prototipo funcione impecable este fin de semana?
- ¿Cómo diseñar la demo perfecta para que los jueces queden alucinados en 2 minutos?

2. TUS 3 RECOMENDACIONES DE IDEAS GANADORAS PARA EL HACKATHON:
Proponnos 3 ideas de nivel mundial para competir por el 1.er puesto ($20,000), con:
- Nombre y Track.
- Arquitectura técnica y por qué aprovecha NVIDIA Nemotron y Nebius.
- Auditoría / Puntuación (1 al 10) y argumento de por qué ganaría frente al resto de participantes.
"""

    print("Enviando petición a nvidia/Nemotron-3_5-Lightning...")
    res = client.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        model="nvidia/Nemotron-3_5-Lightning",
        temperature=0.6,
        max_tokens=3000
    )
    
    print(f"\n[OK] Respuesta generada en {res['latency_seconds']} s ({res['completion_tokens']} tokens generados)\n")
    print(res["content"])
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
