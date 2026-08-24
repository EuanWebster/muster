"""Aggregate every downloaded model across all tools into one list, for the Models tab.

Each record: key (a matchable identifier), source (where it's stored), path,
size_bytes, and used_in (registry tool ids whose live model list references it).
"""
import subprocess
from pathlib import Path

import hf_scan

OLLAMA_STORE_NOTE = "~/.ollama/models (content-addressed blob store, no per-model file path)"

# tool ids whose model_dir is a plain folder of model files worth listing individually
DIR_SCAN_TOOLS = ("comfyui", "swarmui", "ltx-2.5")
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".gguf", ".pt", ".bin")


def _hf_records() -> list[dict]:
    records = []
    for r in hf_scan.scan_gguf_detailed():
        records.append({
            "key": r["key"],
            "name": r["key"],
            "source": "huggingface-cache",
            "path": r["path"],
            "size_bytes": r["size_bytes"],
        })
    return records


def _ollama_records() -> list[dict]:
    records = []
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return records
    for line in out.strip().splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        size_str = None
        for i, tok in enumerate(parts):
            if tok in ("GB", "MB", "KB", "TB") and i > 0:
                size_str = f"{parts[i - 1]} {tok}"
                break
        records.append({
            "key": name,
            "name": name,
            "source": "ollama",
            "path": OLLAMA_STORE_NOTE,
            "size_bytes": None,
            "size_str": size_str,
        })
    return records


def _dir_records(entries: list[dict]) -> list[dict]:
    records = []
    for e in entries:
        if e["id"] not in DIR_SCAN_TOOLS or not e.get("model_dir"):
            continue
        model_dir = Path(e["model_dir"]).expanduser()
        if not model_dir.exists():
            continue
        for f in model_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                records.append({
                    "key": f.name,
                    "name": f.name,
                    "source": e["id"],
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                })
    return records


def _human_size(size_bytes) -> str:
    if not size_bytes:
        return "?"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def gather(entries: list[dict]) -> list[dict]:
    """entries must already have live "models" populated (see app.get_models)."""
    records = _hf_records() + _ollama_records() + _dir_records(entries)
    for r in records:
        r["used_in"] = [e["id"] for e in entries if any(r["key"] in m for m in e.get("models", []))]
        r["size_display"] = r.get("size_str") or _human_size(r.get("size_bytes"))
    records.sort(key=lambda r: (-len(r["used_in"]), r["source"], r["key"]))
    return records
