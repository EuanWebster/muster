"""Detect this machine's GPU/CPU/RAM, so llama.cpp gets the right flags
regardless of whose box this dashboard is running on.

Picks the highest-VRAM GPU as "the" GPU (skips small igpu carve-outs like an
APU's 512MB share, which would never be the one you'd offload a model to).
"""
import functools
import json
import os
import shutil
import subprocess
from pathlib import Path


def _run_json(cmd: list[str]) -> dict:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return json.loads(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def _find_rocm_smi() -> str | None:
    found = shutil.which("rocm-smi")
    if found:
        return found
    for base in sorted(Path("/opt").glob("rocm*")):
        candidate = base / "bin" / "rocm-smi"
        if candidate.exists():
            return str(candidate)
    return None


def _detect_amd_gpu() -> tuple[str, float] | None:
    rocm_smi = _find_rocm_smi()
    if not rocm_smi:
        return None
    mem = _run_json([rocm_smi, "--showmeminfo", "vram", "--json"])
    names = _run_json([rocm_smi, "--showproductname", "--json"])
    if not mem:
        return None
    best_card, best_vram = None, 0
    for card, info in mem.items():
        vram = int(info.get("VRAM Total Memory (B)", 0))
        if vram > best_vram:
            best_card, best_vram = card, vram
    if not best_card:
        return None
    name = names.get(best_card, {}).get("Card Series", "AMD GPU")
    return name, round(best_vram / 1024**3, 1)


def _detect_nvidia_gpu() -> tuple[str, float] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if not lines:
        return None
    best_name, best_vram = None, 0
    for line in lines:
        name, mib = [p.strip() for p in line.rsplit(",", 1)]
        vram_gb = round(int(mib) / 1024, 1)
        if vram_gb > best_vram:
            best_name, best_vram = name, vram_gb
    return (best_name, best_vram) if best_name else None


def _ram_total_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return 0.0


@functools.lru_cache(maxsize=1)
def detect() -> dict:
    """Static per-machine facts - cached for the life of the process."""
    amd_gpu = _detect_amd_gpu()
    gpu_vendor, gpu = ("amd", amd_gpu) if amd_gpu else ("nvidia", _detect_nvidia_gpu())
    if not gpu:
        gpu_vendor = "none"
    gpu_name, vram_gb = gpu if gpu else (None, 0.0)
    return {
        "cpu_cores": os.cpu_count() or 4,
        "ram_total_gb": _ram_total_gb(),
        "gpu_vendor": gpu_vendor,
        "gpu_name": gpu_name,
        "vram_total_gb": vram_gb,
    }


def llama_server_flags(hw: dict) -> list[str]:
    """-ngl (GPU layers to offload) and --threads, from detected hardware.

    ponytail: offloads all layers whenever any GPU is present rather than
    sizing -ngl to the model - llama.cpp itself falls back layers to CPU/RAM
    if VRAM runs out, so this is correct, just not maximally tuned per-model.
    """
    ngl = "999" if hw["gpu_vendor"] != "none" else "0"
    return ["-ngl", ngl, "--threads", str(hw["cpu_cores"])]


if __name__ == "__main__":
    print(json.dumps(detect(), indent=2))
