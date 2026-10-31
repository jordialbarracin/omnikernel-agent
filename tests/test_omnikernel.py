"""
Unit and regression tests for OmniKernel Patterns, Analytical Profiler, and Verifier.
"""

import pytest
from omnikernel.patterns import CANONICAL_PATTERNS, get_pattern, list_patterns
from omnikernel.profiler import KernelProfiler
from omnikernel.verifier import KernelVerifier


def test_canonical_patterns():
    patterns = list_patterns()
    assert len(patterns) >= 4
    ids = [p["id"] for p in patterns]
    assert "fused_attention" in ids
    assert "fp8_quantized_gemm" in ids
    
    pat = get_pattern("fused_attention")
    assert pat is not None
    assert "FlashAttention" in pat["name"]


def test_kernel_profiler_analytical_model():
    prof = KernelProfiler()
    res = prof.profile_pattern("fused_attention", batch_size=2)
    assert res["speedup_factor"] == "3.42x"
    assert "roofline_model" in res
    # Verify the transparency tag is present
    regime = res["roofline_model"]["triton_regime"]
    assert "analytical model estimate" in regime


def test_kernel_verifier_equivalence():
    ver = KernelVerifier()
    res = ver.verify_pattern("fused_attention")
    assert res["passed"] is True
    assert res["status"] == "NUMERIC_EQUIVALENCE_CONFIRMED"
    assert res["max_absolute_error"] < 0.01
    assert res["dynamic_shape_validation"]["all_lengths_passed"] is True
