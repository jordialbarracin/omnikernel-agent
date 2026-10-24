"""
OmniKernel FastAPI Server: Web dashboard backend for interactive GPU kernel synthesis & benchmarking.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from omnikernel.patterns import list_patterns, get_pattern, CANONICAL_PATTERNS
from omnikernel.synthesizer import KernelSynthesizer
from omnikernel.verifier import KernelVerifier
from omnikernel.profiler import KernelProfiler

app = FastAPI(title="OmniKernel API", description="Autonomous GPU Kernel Architect powered by NVIDIA Nemotron & Nebius")

# Static assets directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

class OptimizeRequest(BaseModel):
    pattern_id: Optional[str] = "fused_attention"
    custom_code: Optional[str] = None
    target_gpu: Optional[str] = "NVIDIA Hopper H100 (Tensor Cores)"
    precision: Optional[str] = "FP16"

@app.get("/api/patterns")
def get_patterns():
    return {"patterns": list_patterns()}

@app.get("/api/pattern/{pattern_id}")
def get_pattern_detail(pattern_id: str):
    if pattern_id not in CANONICAL_PATTERNS:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return CANONICAL_PATTERNS[pattern_id]

@app.post("/api/optimize")
def optimize_kernel(req: OptimizeRequest):
    synthesizer = KernelSynthesizer()
    verifier = KernelVerifier()
    profiler = KernelProfiler(target_gpu=req.target_gpu)

    code_target = req.custom_code if req.custom_code else req.pattern_id

    # 1. Synthesis
    synth_res = synthesizer.synthesize(
        code_or_pattern_id=code_target,
        target_gpu=req.target_gpu,
        precision=req.precision
    )

    # 2. Verification
    pattern_key = req.pattern_id if req.pattern_id in CANONICAL_PATTERNS else "generic"
    verif_res = verifier.verify_pattern(pattern_key, precision=req.precision)

    # 3. Profiling & Benchmark
    prof_res = profiler.profile_pattern(pattern_key)

    return {
        "status": "success",
        "synthesis": synth_res,
        "verification": verif_res,
        "profiling": prof_res
    }

@app.get("/api/benchmark/llama")
def benchmark_llama(batch_size: int = 4, seq_len: int = 2048):
    try:
        from benchmark_llama_layer import run_llama_layer_benchmark
        return run_llama_layer_benchmark(batch_size=batch_size, seq_len=seq_len)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "engine": "OmniKernel v1.0",
        "model": "nvidia/Nemotron-3_5-Lightning",
        "cloud_provider": "Nebius Token Factory",
        "verified_templates": 4
    }

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>OmniKernel Dashboard Ready</h1>")

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
