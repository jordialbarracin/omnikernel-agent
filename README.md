# OmniKernel: Autonomous GPU Kernel Architect & Triton Compiler Agent

> **Transparency note:** every performance figure in this document (latencies, speedups, FLOPs/Byte, VRAM) comes from an *analytical roofline model* implemented in `omnikernel/profiler.py` and `benchmark_llama_layer.py`. They are **modeled estimates, not measurements on GPU hardware.** The synthesized Triton kernels are not executed on a GPU in this demo.

**Nebius x NVIDIA Global AI Hackathon**  
*Track:* **Coding and Agentic Engineering Track**

OmniKernel is an autonomous GPU kernel compiler agent powered by **NVIDIA Nemotron-3.5-Lightning** on **Nebius Token Factory**. It analyzes PyTorch operations, maps them to GPU memory hierarchies (SRAM, L2 cache, TMA asynchronous transfers, and Tensor Cores), synthesizes production-ready OpenAI Triton kernels, and includes a numerical-parity verification harness.

---

## Core Capabilities

1. **Hardware-Specific JIT Synthesis (Hopper H100):** Targets `sm_90a` with TMA (`cp.async.bulk.tensor`) and asynchronous Tensor Core matrix instructions (`wgmma.mma_async`).
2. **Dynamic Sequence Length & Boundary Masking:** Handles non-power-of-two dynamic sequence dimensions without memory segmentation faults or online softmax corruption.
3. **FP16 Softmax Numerical Stability:** Safe exponential scaling clamped to prevent `NaN` or `Inf` divergences across extreme logit bounds.
4. **Roofline Model Analysis:** Models arithmetic intensity rising from 18.2 to 164.8 FLOPs/Byte, projecting ~3.42x latency and ~-78% peak VRAM for fused attention (analytical estimates, not hardware measurements).
5. **Verification Harness:** NumPy-based reference comparison (`atol=1e-3`, `rtol=1e-2`); synthesized kernels are not executed on a GPU in this demo.

---

## Analytical Model Estimates (target: NVIDIA H100 SXM5 80GB; not measured)

| Operator Pattern | Baseline Latency | OmniKernel Triton Latency | Speedup | Arithmetic Intensity | VRAM Reduction | Tolerance target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FlashAttention (Fused QK^T V)** | 4.82 ms | **1.41 ms** | **3.42x** | **18.2 $\rightarrow$ 164.8 FLOPs/B** | **-78.4% (1.8GB $\rightarrow$ 390MB)** | target (`atol=1e-3`) |
| **Fused RMSNorm + Residual** | 1.15 ms | **0.38 ms** | **3.02x** | **12.4 $\rightarrow$ 98.1 FLOPs/B** | **-64.2%** | target (`atol=1e-3`) |
| **Fast GELU Activation** | 0.89 ms | **0.31 ms** | **2.87x** | **8.1 $\rightarrow$ 74.5 FLOPs/B** | **-52.0%** | target (`atol=1e-3`) |
| **FP8 Quantized GEMM** | 3.20 ms | **1.05 ms** | **3.05x** | **22.0 $\rightarrow$ 185.2 FLOPs/B** | **-71.5%** | target (`atol=1e-2`) |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
NEBIUS_API_KEY=your_token_factory_key
```

### 3. Launch GPU Workstation
```bash
python -m uvicorn omnikernel.server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.


## Current status & limitations (honest summary)
- **Real:** Nemotron-3.5-Lightning (Nebius Token Factory) synthesizes Triton kernels from PyTorch operators; the web console, pattern library and roofline math run end-to-end.
- **Modeled, not measured:** all latency / speedup / VRAM / arithmetic-intensity numbers come from an analytical model (hard-coded per-pattern constants plus a throughput formula). No kernel was benchmarked on an H100 (development machine: GeForce RTX 2060).
- **Verifier:** `verifier.py` compares a NumPy reference implementation against a NumPy-perturbed copy; it does **not** execute the synthesized Triton kernels. Reported error values are illustrative.
- **Fallback:** if the API is unreachable, the tool returns pre-written kernel templates (flagged `is_fallback`). The PTX/SASS tab shows template text, not compiler output.
- **Next step:** run the generated kernels on real hardware and replace modeled numbers with measured ones.
