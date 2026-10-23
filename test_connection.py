import sys
import os
from config import validate_config, NEBIUS_API_KEY, DEFAULT_MODEL, NEBIUS_BASE_URL, ENV_PATH
from nebius_client import NebiusNvidiaClient

def main():
    print("=" * 65)
    print("  Nebius Token Factory & NVIDIA Model Connectivity Test")
    print("=" * 65)
    print(f"Base URL:      {NEBIUS_BASE_URL}")
    print(f"Target Model:  {DEFAULT_MODEL}")
    
    if not validate_config():
        print("\n[!] AVISO: No se ha detectado la clave NEBIUS_API_KEY en el archivo .env.")
        print("    Para configurarla:")
        print("    1. Abre el archivo .env en esta carpeta:")
        print(f"       {ENV_PATH}")
        print("    2. Añade tu API Key:")
        print("       NEBIUS_API_KEY=tu_clave_de_nebius_aqui")
        print("    3. Vuelve a ejecutar este script: python test_connection.py\n")
        sys.exit(1)
        
    print(f"API Key:       {NEBIUS_API_KEY[:8]}...{NEBIUS_API_KEY[-4:] if len(NEBIUS_API_KEY) > 12 else ''}")
    print("\n[+] Conectando con Nebius Token Factory...")
    
    try:
        client = NebiusNvidiaClient()
        print("[+] Enviando prompt de prueba a NVIDIA Nemotron 70B...")
        
        prompt = "Hello! Please confirm that NVIDIA Nemotron-70B is active and responding on Nebius Token Factory in one concise sentence."
        res = client.complete(
            prompt=prompt,
            system_prompt="You are an advanced NVIDIA AI model running on Nebius high-performance infrastructure."
        )
        
        print("\n" + "-" * 65)
        print("Respuesta recibida:")
        print(f"'{res['content'].strip()}'")
        print("-" * 65)
        print(f"[OK] Latencia total:      {res['latency_seconds']} s")
        print(f"[OK] Tokens de respuesta: {res['completion_tokens']} tokens")
        print(f"[OK] Modelo confirmado:   {res['model']}")
        print("=" * 65)
        print("¡Conexion verificada al 100%! El entorno esta listo para el hackathon.")
        print("=" * 65)
        
    except Exception as e:
        print(f"\n[X] Error durante la llamada a la API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
