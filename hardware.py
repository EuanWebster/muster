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
    card = _amd_best_card()
    if not card:
        return None
    mem = _run_json([_find_rocm_smi(), "--showmeminfo", "vram", "--json"])
    names = _run_json([_find_rocm_smi(), "--showproductname", "--json"])
    vram = int(mem.get(card, {}).get("VRAM Total Memory (B)", 0))
    name = names.get(card, {}).get("Card Series", "AMD GPU")
    return name, round(vram / 1024**3, 1)


@functools.lru_cache(maxsize=1)
def _amd_best_card() -> str | None:
    """Same highest-VRAM card detect() picks, so live usage tracks the same GPU."""
    rocm_smi = _find_rocm_smi()
    if not rocm_smi:
        return None
    mem = _run_json([rocm_smi, "--showmeminfo", "vram", "--json"])
    best_card, best_vram = None, 0
    for card, info in mem.items():
        vram = int(info.get("VRAM Total Memory (B)", 0))
        if vram > best_vram:
            best_card, best_vram = card, vram
    return best_card


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


def _ram_used_gb() -> float:
    """Total minus MemAvailable (accounts for reclaimable cache, unlike MemFree)."""
    try:
        fields = {}
        with open("/proc/meminfo") as f:
            for line in f:
                for key in ("MemTotal:", "MemAvailable:"):
                    if line.startswith(key):
                        fields[key] = int(line.split()[1])
        if "MemTotal:" in fields and "MemAvailable:" in fields:
            return round((fields["MemTotal:"] - fields["MemAvailable:"]) / 1024**2, 1)
    except OSError:
        pass
    return 0.0


def _cpu_load_pct(cores: int) -> float:
    try:
        return round(os.getloadavg()[0] / cores * 100, 1)
    except OSError:
        return 0.0


def _amd_gpu_live() -> tuple[float, float] | None:
    """(vram_used_gb, gpu_util_pct) for the same card detect() picked, or None."""
    card = _amd_best_card()
    rocm_smi = _find_rocm_smi()
    if not card or not rocm_smi:
        return None
    mem = _run_json([rocm_smi, "--showmeminfo", "vram", "--json"])
    use = _run_json([rocm_smi, "--showuse", "--json"])
    used_b = int(mem.get(card, {}).get("VRAM Total Used Memory (B)", 0))
    util = float(use.get(card, {}).get("GPU use (%)", 0))
    return round(used_b / 1024**3, 1), util


def _nvidia_gpu_live() -> tuple[float, float] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    line = next((l.strip() for l in r.stdout.splitlines() if l.strip()), None)
    if not line:
        return None
    used_mib, util = [p.strip() for p in line.split(",")]
    return round(int(used_mib) / 1024, 1), float(util)


@functools.lru_cache(maxsize=1)
def _physical_cores() -> int:
    """Physical cores, not SMT threads.

    llama.cpp decode is memory-bandwidth bound, so SMT siblings contend rather than
    add throughput: on the 7800X3D (8c/16t), --threads 16 measured 7.21 tok/s against
    10.03 at --threads 8 on Qwen3.8-Flash-Next - a 39% loss. os.cpu_count() returns
    logical CPUs, so count distinct core ids instead.
    """
    try:
        ids = set()
        phys = pkg = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("core id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("physical id"):
                pkg = line.split(":")[1].strip()
            elif not line.strip() and phys is not None:
                ids.add((pkg, phys)); phys = pkg = None
        if phys is not None:
            ids.add((pkg, phys))
        if ids:
            return len(ids)
    except OSError:
        pass
    n = os.cpu_count() or 4
    return max(1, n // 2)


def detect() -> dict:
    """Static totals (name/cores/RAM/VRAM capacity) plus current usage.

    The totals barely change call to call (only cores is actually cached);
    usage is read fresh every call, so callers polling this get a live view.
    """
    amd_gpu = _detect_amd_gpu()
    gpu_vendor, gpu = ("amd", amd_gpu) if amd_gpu else ("nvidia", _detect_nvidia_gpu())
    if not gpu:
        gpu_vendor = "none"
    gpu_name, vram_gb = gpu if gpu else (None, 0.0)

    live = None
    if gpu_vendor == "amd":
        live = _amd_gpu_live()
    elif gpu_vendor == "nvidia":
        live = _nvidia_gpu_live()
    vram_used_gb, gpu_util_pct = live if live else (0.0, 0.0)

    cores = _physical_cores()
    return {
        "cpu_cores": cores,
        "ram_total_gb": _ram_total_gb(),
        "ram_used_gb": _ram_used_gb(),
        "cpu_load_pct": _cpu_load_pct(cores),
        "gpu_vendor": gpu_vendor,
        "gpu_name": gpu_name,
        "vram_total_gb": vram_gb,
        "vram_used_gb": vram_used_gb,
        "gpu_util_pct": gpu_util_pct,
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
