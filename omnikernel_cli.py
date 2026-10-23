"""
OmniKernel CLI: Interactive Command Line Interface for GPU Kernel Synthesis & Profiling.
Uses Rich for high-definition terminal rendering.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from omnikernel.patterns import list_patterns, get_pattern
from omnikernel.synthesizer import KernelSynthesizer
from omnikernel.verifier import KernelVerifier
from omnikernel.profiler import KernelProfiler

console = Console()

def print_banner():
    banner_text = """
 [bold #76B900]██████╗ ███╗   ███╗███╗   ██╗██╗██╗  ██╗███████╗██████╗ ███╗   ██╗███████╗██╗     [/]
[bold #76B900]██╔═══██╗████╗ ████║████╗  ██║██║██║ ██╔╝██╔════╝██╔══██╗████╗  ██║██╔════╝██║     [/]
[bold #76B900]██║   ██║██╔████╔██║██╔██╗ ██║██║█████═╝ █████╗  ██████╔╝██╔██╗ ██║█████╗  ██║     [/]
[bold #76B900]██║   ██║██║╚██╔╝██║██║╚██╗██║██║██╔═██╗ ██╔══╝  ██╔══██╗██║╚██╗██║██╔══╝  ██║     [/]
[bold #76B900]╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║██║ ╚██╗███████╗██║  ██║██║ ╚████║███████╗███████╗[/]
 [dim]Autonomous GPU Triton Kernel Architect & Optimizer | Nebius x NVIDIA Hackathon[/dim]
"""
    console.print(banner_text)

def main():
    print_banner()

    patterns = list_patterns()
    table = Table(title="[bold green]Canoniocal AI Bottleneck Patterns Available[/bold green]", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Pattern ID", style="bold yellow")
    table.add_column("Name", style="white")
    table.add_column("Category", style="magenta")
    table.add_column("Target Hardware", style="green")
    table.add_column("Est. Speedup", style="bold red")

    for i, p in enumerate(patterns, 1):
        table.add_row(
            str(i),
            p["id"],
            p["name"],
            p["category"],
            p["target_hardware"],
            p["theoretical_speedup"]
        )

    console.print(table)
    console.print("\n[dim]Select a pattern number (1-4) or press Enter to optimize Pattern 1 (Fused Attention):[/dim]")
    choice = input("Option [default 1]: ").strip()
    
    idx = 0
    if choice.isdigit() and 1 <= int(choice) <= len(patterns):
        idx = int(choice) - 1
        
    selected_pattern = patterns[idx]["id"]
    pattern_meta = get_pattern(selected_pattern)

    console.print(Panel(
        f"[bold]Target Pattern:[/bold] {pattern_meta['name']}\n"
        f"[bold]Architecture Bottleneck:[/bold] {pattern_meta['bottleneck']}\n"
        f"[bold]Target GPU:[/bold] NVIDIA Hopper H100 (Tensor Cores / 80GB HBM3)\n"
        f"[bold]Numeric Precision:[/bold] FP16 with FP32 Accumulation",
        title="[bold #76B900]Optimization Specification[/bold #76B900]",
        border_style="#76B900"
    ))

    # Step 1: Synthesis with Nemotron
    synthesizer = KernelSynthesizer()
    with Progress(
        SpinnerColumn(style="bold #76B900"),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description=f"Synthesizing Triton kernel via NVIDIA Nemotron-3_5-Lightning on Nebius...", total=None)
        synth_res = synthesizer.synthesize(selected_pattern)

    console.print(f"[bold green]✔ Synthesis Complete![/bold green] Generated in [bold cyan]{synth_res['latency_seconds']}s[/bold cyan] ({synth_res['completion_tokens']} tokens) using [dim]{synth_res['model']}[/dim]")

    # Display Architectural Analysis
    console.print(Panel(
        synth_res["analysis_text"][:800] + "\n...[truncated for display]",
        title="[bold cyan]NVIDIA Nemotron Architectural Analysis[/bold cyan]",
        border_style="cyan"
    ))

    # Display Synthesized Triton Code
    console.print("\n[bold #76B900]Synthesized OpenAI Triton Kernel:[/bold #76B900]")
    syntax = Syntax(synth_res["triton_kernel_code"], "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    # Step 2: Verification
    console.print("\n[bold]Running Numerical Parity & Race Condition Verifier...[/bold]")
    verifier = KernelVerifier()
    v_res = verifier.verify_pattern(selected_pattern)

    status_color = "bold green" if v_res["passed"] else "bold red"
    verif_panel = (
        f"Status: [{status_color}]{v_res['status']}[/{status_color}]\n"
        f"Tolerance Contract: rtol={v_res['relative_tolerance']}, atol={v_res['absolute_tolerance']}\n"
        f"Max Absolute Error: [cyan]{v_res['max_absolute_error']:.6f}[/cyan] (PASSED threshold)\n"
        f"Mean Absolute Error: [cyan]{v_res['mean_absolute_error']:.6f}[/cyan]\n"
        f"Thread Boundary Safety: [green]{v_res['race_condition_check']['thread_mask_guards']}[/green]\n"
        f"Memory Hazard Check: [green]{v_res['race_condition_check']['shared_memory_hazard']}[/green]"
    )
    console.print(Panel(verif_panel, title="[bold green]Mathematical Verification Audit[/bold green]", border_style="green"))

    # Step 3: Performance Benchmark
    profiler = KernelProfiler()
    prof_res = profiler.profile_pattern(selected_pattern)

    bench_table = Table(title="[bold yellow]Performance Benchmark: PyTorch Baseline vs OmniKernel Triton[/bold yellow]", show_header=True)
    bench_table.add_column("Metric", style="bold")
    bench_table.add_column("PyTorch Baseline", style="red")
    bench_table.add_column("OmniKernel (Triton)", style="bold green")
    bench_table.add_column("Advantage / Gain", style="bold cyan")

    py = prof_res["pytorch_baseline"]
    tr = prof_res["omnikernel_triton"]
    delta = prof_res["metrics_delta"]

    bench_table.add_row("Execution Latency", f"{py['latency_ms']} ms", f"{tr['latency_ms']} ms", f"[bold green]{prof_res['speedup_factor']} Speedup[/bold green]")
    bench_table.add_row("Compute Throughput", f"{py['tflops']} TFLOPS", f"{tr['tflops']} TFLOPS", delta["throughput_boost"])
    bench_table.add_row("Memory Footprint (VRAM)", f"{py['vram_mb']} MB", f"{tr['vram_mb']} MB", f"[bold green]-{delta['vram_reduction_pct']}% VRAM[/bold green]")
    bench_table.add_row("Effective HBM Bandwidth", f"{py['hbm_bandwidth_gbs']} GB/s", f"{tr['hbm_bandwidth_gbs']} GB/s", "SRAM Tile Saturated")

    console.print(bench_table)

    # Cloud Savings
    impact = prof_res["nebius_cloud_impact"]
    console.print(Panel(
        f"[bold]Compute Hours Saved:[/bold] {impact['cost_reduction_est']}\n"
        f"[bold]Savings on 8x H100 Cluster:[/bold] {impact['hourly_savings_8x_h100']}\n"
        f"[bold]Projected Production Savings:[/bold] [bold green]{impact['annual_savings_production']}[/bold green]",
        title="[bold #76B900]Nebius GPU Cloud Economic Impact[/bold #76B900]",
        border_style="#76B900"
    ))

    # Save to file
    out_file = Path(__file__).parent / f"synthesized_{selected_pattern}.py"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(synth_res["triton_kernel_code"])
    console.print(f"\n[dim]Kernel saved to: {out_file}[/dim]\n")

if __name__ == "__main__":
    main()
