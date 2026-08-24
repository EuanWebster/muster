"""Find installed things not yet in the registry. Never writes - just reports candidates.

Deliberately does NOT scan all systemd units - on a normal desktop that's 40+
GNOME/desktop-portal/hardware services with zero signal for "new LLM tool
installed". Docker containers, ~/.local/bin executables, and listening ports
are a much higher-signal set for this use case.
skipped: systemd-unit scanning, add back with an allowlist/keyword filter if
a future tool installs as a systemd unit and isn't caught by the other signals.
"""
import subprocess
from pathlib import Path


def _known_containers(entries: list[dict]) -> set[str]:
    return {e["container"] for e in entries if e.get("container")}


def _known_bin_paths(entries: list[dict]) -> set[str]:
    known = set()
    for e in entries:
        for p in e.get("install_paths", []):
            known.add(str(Path(p).expanduser()))
    return known


def find_candidates(entries: list[dict]) -> list[dict]:
    candidates = []

    known_containers = _known_containers(entries)
    try:
        r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}"],
                            capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if not line.strip():
                continue
            name, image = line.split("\t", 1)
            if name not in known_containers:
                candidates.append({"type": "docker-container", "name": name, "image": image})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    known_bins = _known_bin_paths(entries)
    local_bin = Path.home() / ".local/bin"
    if local_bin.exists():
        for exe in local_bin.iterdir():
            if str(exe) not in known_bins and exe.is_file():
                candidates.append({"type": "executable", "name": exe.name, "path": str(exe)})

    return candidates
