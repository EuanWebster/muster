"""Scan the local HuggingFace cache for downloaded GGUF quants.

Ported from ~/Projects/scutwork/server.py's _scan_local_quants() - Unsloth
Studio's /v1/models only reports the last-active quant per repo, so this
reads the cache directly to see everything actually downloaded.
"""
from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"


def scan_gguf_detailed(cache_dir: Path = HF_CACHE) -> list[dict]:
    """One record per downloaded GGUF file: repo, quant, key (repo:quant), path, size_bytes."""
    records = []
    if not cache_dir.exists():
        return records
    for repo_dir in cache_dir.iterdir():
        if not repo_dir.name.startswith("models--"):
            continue
        repo_id = repo_dir.name.removeprefix("models--").replace("--", "/", 1)
        short_name = repo_id.split("/")[-1].removesuffix("-GGUF")
        for gguf in repo_dir.glob("snapshots/*/*.gguf"):
            stem = gguf.stem
            if "mmproj" in stem.lower():
                continue
            quant = stem.removeprefix(f"{short_name}-")
            resolved = gguf.resolve()  # HF cache files are usually symlinks into blobs/
            records.append({
                "repo": repo_id,
                "quant": quant,
                "key": f"{repo_id}:{quant}",
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size if resolved.exists() else None,
            })
    return records


def scan_gguf(cache_dir: Path = HF_CACHE) -> dict[str, list[str]]:
    quants: dict[str, list[str]] = {}
    for r in scan_gguf_detailed(cache_dir):
        quants.setdefault(r["repo"], []).append(r["quant"])
    return quants


if __name__ == "__main__":
    for repo, qs in sorted(scan_gguf().items()):
        for q in sorted(qs):
            print(f"{repo}:{q}")
