"""Scan the local HuggingFace cache for downloaded GGUF quants.

Ported from ~/Projects/scutwork/server.py's _scan_local_quants() - Unsloth
Studio's /v1/models only reports the last-active quant per repo, so this
reads the cache directly to see everything actually downloaded.
"""
import re
from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"


SHARD_RE = re.compile(r"-(\d{5})-of-\d{5}$")


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
        # Big models are sharded into "-00001-of-000NN" parts and often sit one dir
        # deeper (snapshots/<hash>/<QUANT>/), so glob recursively. llama.cpp is given
        # only the first shard and finds the rest itself, so skip 00002+ and report
        # the summed size of the whole set.
        for gguf in sorted(repo_dir.glob("snapshots/*/**/*.gguf")):
            stem = gguf.stem
            if "mmproj" in stem.lower():
                continue
            shard = SHARD_RE.search(stem)
            if shard and shard.group(1) != "00001":
                continue
            quant = SHARD_RE.sub("", stem).removeprefix(f"{short_name}-")
            # Report the snapshot path, NOT the resolved blob path: sharded models need
            # the "-00001-of-000NN.gguf" name so llama.cpp can find the sibling shards
            # (a blob hash gives "invalid split file name"). llama.cpp follows symlinks.
            resolved = gguf.resolve()  # only for size/existence checks
            if shard:
                size = sum(p.resolve().stat().st_size
                           for p in gguf.parent.glob(SHARD_RE.sub("-*", stem) + ".gguf")
                           if p.resolve().exists())
            else:
                size = resolved.stat().st_size if resolved.exists() else None
            records.append({
                "repo": repo_id,
                "quant": quant,
                "key": f"{repo_id}:{quant}",
                "path": str(gguf),
                "size_bytes": size,
            })
    # A repo can hold several snapshot revisions of the same quant, and older ones
    # often have dangling symlinks. Keep one record per key, preferring a resolvable file.
    best: dict[str, dict] = {}
    for r in records:
        cur = best.get(r["key"])
        if cur is None or (not Path(cur["path"]).exists() and Path(r["path"]).exists()):
            best[r["key"]] = r
    return list(best.values())


def scan_gguf(cache_dir: Path = HF_CACHE) -> dict[str, list[str]]:
    quants: dict[str, list[str]] = {}
    for r in scan_gguf_detailed(cache_dir):
        quants.setdefault(r["repo"], []).append(r["quant"])
    return quants


if __name__ == "__main__":
    for repo, qs in sorted(scan_gguf().items()):
        for q in sorted(qs):
            print(f"{repo}:{q}")
