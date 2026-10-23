"""
Canonical AI GPU Bottleneck Patterns for OmniKernel.
These represent high-impact architectural bottlenecks in modern LLMs & vision models.
"""

from typing import Dict, Any, List

CANONICAL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "fused_attention": {
        "id": "fused_attention",
        "name": "Fused Scaled Dot-Product Attention (FlashAttention Pattern)",
        "category": "Transformer Core / Memory Bound",
        "target_hardware": "NVIDIA H100 / A100 / RTX 4090 (Hopper / Ampere Tensor Cores)",
        "description": "Fuses Q @ K^T, online Softmax scaling, and @ V into a single GPU pass without writing the O(N^2) attention matrix to slow HBM.",
        "bottleneck": "Standard PyTorch materializes the full [Batch, Heads, SeqLen, SeqLen] attention weight tensor in high-bandwidth memory (HBM), causing huge VRAM consumption and high memory latency.",
        "pytorch_code": '''import torch
import math

def pytorch_attention_baseline(q, k, v):
    """
    Standard unfused PyTorch Scaled Dot-Product Attention.
    q, k, v: [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    # Step 1: Compute attention scores (Writes full N x N matrix to HBM)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Step 2: Softmax over sequence dimension (HBM read + write)
    p = torch.softmax(scores, dim=-1)
    
    # Step 3: Context projection (HBM read + write)
    out = torch.matmul(p, v)
    return out
''',
        "sample_dims": {
            "BATCH": 2,
            "NUM_HEADS": 8,
            "SEQ_LEN": 2048,
            "HEAD_DIM": 64
        },
        "theoretical_speedup": "3.2x - 4.1x",
        "memory_saving": "78% VRAM reduction (eliminates O(N^2) intermediate buffer)"
    },

    "fused_mlp_norm": {
        "id": "fused_mlp_norm",
        "name": "Fused LayerNorm + GELU + Linear (Fused Transformer MLP Block)",
        "category": "Feed-Forward / Elementwise Fusion",
        "target_hardware": "NVIDIA Hopper / Ada Lovelace / Ampere",
        "description": "Combines Layer Normalization, Gaussian Error Linear Unit (GELU) activation, and linear projection into a fused kernel utilizing SRAM cache.",
        "bottleneck": "Unfused execution requires multiple sequential CUDA kernel launches with intermediate reads/writes to VRAM for normalized outputs and activations before GEMM.",
        "pytorch_code": '''import torch
import torch.nn.functional as F

def pytorch_fused_mlp_baseline(x, weight, bias, gamma, beta, eps=1e-5):
    """
    Standard sequential PyTorch Transformer LayerNorm + GELU + Projection.
    x: [BATCH_SIZE, HIDDEN_DIM]
    weight: [OUT_DIM, HIDDEN_DIM]
    """
    # Step 1: LayerNorm (HBM read -> compute mean/var -> HBM write)
    normed = F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=gamma, bias=beta, eps=eps)
    
    # Step 2: GELU Non-linearity (HBM read -> compute -> HBM write)
    activated = F.gelu(normed, approximate='tanh')
    
    # Step 3: Linear GEMM (HBM read -> Tensor Core MMA -> HBM write)
    out = F.linear(activated, weight, bias)
    return out
''',
        "sample_dims": {
            "BATCH_SIZE": 1024,
            "HIDDEN_DIM": 4096,
            "OUT_DIM": 4096
        },
        "theoretical_speedup": "2.4x - 3.1x",
        "memory_saving": "54% VRAM reduction (intermediates kept in register/SRAM)"
    },

    "fp8_quantized_gemm": {
        "id": "fp8_quantized_gemm",
        "name": "FP8 Quantized GEMM with On-The-Fly Dequantization",
        "category": "Quantization / Tensor Cores",
        "target_hardware": "NVIDIA H100 Hopper / L40S / RTX 4090 (Ada)",
        "description": "Matrix multiplication decoding 8-bit floating point (E4M3 / E5M2) weights directly into FP32 accumulators on Tensor Cores with zero intermediate memory writes.",
        "bottleneck": "Naïve quantization dequantizes FP8 matrices into FP16/FP32 in VRAM before calling GEMM, causing memory spikes and defeating the throughput advantage of FP8.",
        "pytorch_code": '''import torch

def pytorch_fp8_gemm_baseline(x, weight_fp8, scale_x, scale_w):
    """
    Naïve PyTorch FP8 dequantize followed by FP16/FP32 GEMM.
    x: [M, K] in FP16
    weight_fp8: [N, K] stored in int8/float8 format
    """
    # Step 1: Dequantize weight into high precision memory
    w_fp16 = weight_fp8.to(torch.float16) * scale_w
    
    # Step 2: Scale input
    x_scaled = x * scale_x
    
    # Step 3: Standard GEMM
    out = torch.matmul(x_scaled, w_fp16.t())
    return out
''',
        "sample_dims": {
            "M": 2048,
            "K": 8192,
            "N": 8192
        },
        "theoretical_speedup": "2.8x - 3.8x",
        "memory_saving": "50% memory bandwidth saved via 8-bit memory bus transfers"
    },

    "moe_routing_scatter": {
        "id": "moe_routing_scatter",
        "name": "Top-K Mixture-of-Experts (MoE) Fused Dispatcher & Kernel",
        "category": "MoE Architecture / Irregular Access",
        "target_hardware": "NVIDIA H100 / A100 Clustered Systems",
        "description": "Fuses top-k routing gate evaluation, token sorting, and expert dispatch without expensive torch.index_select memory allocations.",
        "bottleneck": "Sparse MoE models spend up to 35% of total layer time just gathering tokens and scattering them back from expert feedforward blocks.",
        "pytorch_code": '''import torch
import torch.nn.functional as F

def pytorch_moe_routing_baseline(hidden_states, gate_weights, top_k=2):
    """
    Standard PyTorch Top-K MoE Gating & Token Routing.
    hidden_states: [NUM_TOKENS, HIDDEN_DIM]
    gate_weights: [HIDDEN_DIM, NUM_EXPERTS]
    """
    # Compute gate logits
    logits = torch.matmul(hidden_states, gate_weights)
    
    # Softmax over experts
    probs = F.softmax(logits, dim=-1)
    
    # Top-K selection
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    
    # Re-normalize weights
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return topk_probs, topk_indices
''',
        "sample_dims": {
            "NUM_TOKENS": 4096,
            "HIDDEN_DIM": 2048,
            "NUM_EXPERTS": 8
        },
        "theoretical_speedup": "2.2x - 2.9x",
        "memory_saving": "40% VRAM allocation reduction"
    }
}

def get_pattern(pattern_id: str) -> Dict[str, Any]:
    return CANONICAL_PATTERNS.get(pattern_id, CANONICAL_PATTERNS["fused_attention"])

def list_patterns() -> List[Dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "target_hardware": p["target_hardware"],
            "description": p["description"],
            "theoretical_speedup": p["theoretical_speedup"],
            "memory_saving": p["memory_saving"]
        }
        for p in CANONICAL_PATTERNS.values()
    ]
