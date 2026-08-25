"""Local dashboard for managing installed LLM tools/harnesses.

Single-file Flask app. Registry (registry.json) is the source of truth for
what tools exist; status is always checked live, never cached.
"""
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import agent_models
import discover
import hardware
import hf_scan
import model_inventory
import obsidian_sync
import query

BASE = Path(__file__).parent
REGISTRY_FILE = BASE / "registry.json"

app = Flask(__name__, static_folder="static")


def load_registry() -> list[dict]:
    return json.loads(REGISTRY_FILE.read_text())


def save_registry(entries: list[dict]) -> None:
    REGISTRY_FILE.write_text(json.dumps(entries, indent=2) + "\n")


def expand(path: str) -> str:
    return str(Path(path).expanduser())


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    if not port:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host if host not in (None, "0.0.0.0") else "127.0.0.1", port)) == 0
    except OSError:
        return False


def check_status(entry: dict) -> str:
    """Returns 'running' | 'stopped' | 'n/a' | 'unknown'."""
    kind = entry.get("kind")
    try:
        if kind == "systemd-system":
            r = subprocess.run(["systemctl", "is-active", entry["unit"]], capture_output=True, text=True)
            return "running" if r.stdout.strip() == "active" else "stopped"
        if kind == "systemd-user":
            r = subprocess.run(["systemctl", "--user", "is-active", entry["unit"]], capture_output=True, text=True)
            return "running" if r.stdout.strip() == "active" else "stopped"
        if kind == "docker-compose":
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", entry["container"]],
                                capture_output=True, text=True)
            return "running" if r.stdout.strip() == "true" else "stopped"
        if kind == "docker-container":
            r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", entry["container"]],
                                capture_output=True, text=True)
            if r.returncode != 0:
                return "stopped"  # container doesn't exist yet, e.g. portainer
            return "running" if r.stdout.strip() == "true" else "stopped"
        if kind in ("process", "llama-server"):
            cmd = entry.get("status_cmd")
            if cmd:
                r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
                return "running" if r.returncode == 0 and r.stdout.strip() else "stopped"
            return "unknown"
        if kind == "manual":
            return "n/a"
    except FileNotFoundError:
        return "unknown"
    return "unknown"


def capabilities(entry: dict) -> dict:
    """Whether Start/Stop have any real action for this entry - mirrors the
    dispatch logic in api_start/api_stop exactly, without running anything.
    Lets the UI hide/replace buttons that would just error on click."""
    kind = entry.get("kind")
    can_start = kind in ("systemd-system", "systemd-user", "docker-compose", "docker-container", "llama-server") or (
        kind == "process" and bool(entry.get("start_cmd"))
    )
    can_stop = kind in ("systemd-system", "systemd-user", "docker-compose", "docker-container") or (
        kind in ("process", "llama-server") and bool(entry.get("stop_cmd") or entry.get("status_cmd"))
    )
    return {"can_start": can_start, "can_stop": can_stop}


def check_boot_enabled(entry: dict) -> str | None:
    """Returns a short human label if this tool is set to start at boot/login, else None.

    Only meaningful for kinds with a real boot-start mechanism (systemd unit,
    Docker restart policy) - process/manual entries have no generic equivalent
    (they'd need their own separate systemd unit, which isn't modeled here).
    """
    kind = entry.get("kind")
    try:
        if kind == "systemd-system":
            r = subprocess.run(["systemctl", "is-enabled", entry["unit"]], capture_output=True, text=True)
            state = r.stdout.strip()
            return f"enabled at boot ({state})" if state in ("enabled", "static", "enabled-runtime") else None
        if kind == "systemd-user":
            r = subprocess.run(["systemctl", "--user", "is-enabled", entry["unit"]], capture_output=True, text=True)
            state = r.stdout.strip()
            return f"enabled at login ({state})" if state in ("enabled", "static", "enabled-runtime") else None
        if kind in ("docker-compose", "docker-container"):
            r = subprocess.run(["docker", "inspect", "-f", "{{.HostConfig.RestartPolicy.Name}}", entry["container"]],
                                capture_output=True, text=True)
            policy = r.stdout.strip()
            if policy in ("always", "unless-stopped"):
                return f"restarts with Docker ({policy})"
            return None
    except FileNotFoundError:
        return None
    return None


def stop_by_pattern(entry: dict) -> None:
    """Best-effort kill by the same pattern status_cmd uses to detect it's running."""
    pattern = entry["status_cmd"].replace("pgrep -f ", "").strip("'\"")
    subprocess.run(["pkill", "-f", pattern])


def run_version_cmd(cmd: str | None, timeout: int = 20) -> str | None:
    """Runs a version-check command (shell, so it can pipe/awk) and returns the
    last non-empty line of stdout, trimmed - or None if it fails/times out."""
    if not cmd:
        return None
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=Path.home())
    except (subprocess.TimeoutExpired, OSError):
        return None
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else None


def get_models(entry: dict) -> list[str]:
    scan = entry.get("model_scan")
    if scan == "ollama-cli":
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().splitlines()[1:]  # skip header
            return [line.split()[0] for line in lines if line.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
    if scan == "hf-cache-gguf":
        quants = hf_scan.scan_gguf()
        return [f"{repo}:{q}" for repo, qs in sorted(quants.items()) for q in sorted(qs)]
    if scan == "dir-glob":
        model_dir = Path(expand(entry.get("model_dir", "")))
        if not model_dir.exists():
            return []
        return sorted({p.parent.name for p in model_dir.rglob("*") if p.is_file()})[:20]
    if scan == "dsh-config":
        return agent_models.dsh_models()
    if scan == "hermes-config":
        return agent_models.hermes_models()
    if scan == "ollama-integration":
        return agent_models.ollama_integration_models(entry.get("integration_key", entry["id"]))
    return []


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/hardware")
def api_hardware():
    return jsonify(hardware.detect())


@app.route("/api/tools")
def api_tools():
    entries = load_registry()
    for e in entries:
        e["status"] = check_status(e)
        e["port_open"] = port_open(e.get("host") or "127.0.0.1", e.get("port")) if e.get("port") else None
        e["models"] = get_models(e)
        e["boot_enabled"] = check_boot_enabled(e)
        e.update(capabilities(e))
    return jsonify(entries)


@app.route("/api/models")
def api_models():
    entries = load_registry()
    for e in entries:
        e["models"] = get_models(e)
    return jsonify(model_inventory.gather(entries))


@app.route("/api/tools/<tool_id>/start", methods=["POST"])
def api_start(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    kind = entry["kind"]
    body = request.get_json(silent=True) or {}
    try:
        if kind == "systemd-system":
            subprocess.run(["sudo", "systemctl", "start", entry["unit"]], check=True, timeout=15)
        elif kind == "systemd-user":
            subprocess.run(["systemctl", "--user", "start", entry["unit"]], check=True, timeout=15)
        elif kind == "docker-compose":
            subprocess.run(["docker", "compose", "-f", expand(entry["compose_file"]), "up", "-d", entry["service"]],
                            check=True, timeout=60)
        elif kind == "docker-container":
            subprocess.run(["docker", "start", entry["container"]], check=True, timeout=30)
        elif kind == "process" and entry.get("start_cmd"):
            proc = subprocess.Popen(entry["start_cmd"], shell=True, cwd=Path.home(),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
            # Most launch failures (missing deps, wrong path, bad venv) surface within
            # a couple seconds - catch those and report them instead of failing silently.
            # A real server that's still booting after this is assumed to be fine.
            time.sleep(2)
            if proc.poll() is not None:
                output = proc.stdout.read()[-2000:]
                return jsonify({"error": f"Process exited immediately (code {proc.returncode}):\n{output}"}), 500
        elif kind == "llama-server":
            model_key = body.get("model")
            if not model_key:
                return jsonify({"error": "pick a model first"}), 400
            record = next((r for r in hf_scan.scan_gguf_detailed() if r["key"] == model_key), None)
            if not record:
                return jsonify({"error": f"model not found in HF cache: {model_key}"}), 404
            binary = expand(entry["binary"])
            if not Path(binary).exists():
                return jsonify({"error": f"llama-server binary not found: {binary}"}), 500
            # Switching models: a second llama-server can't bind the same port, so
            # replace the running one instead of leaving it serving the old model.
            if check_status(entry) == "running":
                stop_by_pattern(entry)
                for _ in range(20):
                    if not port_open(entry.get("host") or "127.0.0.1", entry["port"]):
                        break
                    time.sleep(0.25)
            cmd = [binary, "-m", record["path"], "--alias", model_key, "--host", entry.get("host", "127.0.0.1"),
                   "--port", str(entry["port"])] + hardware.llama_server_flags(hardware.detect())
            proc = subprocess.Popen(cmd, cwd=Path.home(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
            time.sleep(2)
            if proc.poll() is not None:
                output = proc.stdout.read()[-2000:]
                return jsonify({"error": f"Process exited immediately (code {proc.returncode}):\n{output}"}), 500
        else:
            return jsonify({"error": f"no start action for kind={kind}"}), 400
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify({"ok": True})


@app.route("/api/tools/<tool_id>/stop", methods=["POST"])
def api_stop(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    kind = entry["kind"]
    try:
        if kind == "systemd-system":
            subprocess.run(["sudo", "systemctl", "stop", entry["unit"]], check=True, timeout=15)
        elif kind == "systemd-user":
            subprocess.run(["systemctl", "--user", "stop", entry["unit"]], check=True, timeout=15)
        elif kind == "docker-compose" or kind == "docker-container":
            subprocess.run(["docker", "stop", entry["container"]], check=True, timeout=30)
        elif kind in ("process", "llama-server") and entry.get("stop_cmd"):
            subprocess.run(shlex.split(entry["stop_cmd"]), check=True, timeout=15)
        elif kind in ("process", "llama-server") and entry.get("status_cmd"):
            stop_by_pattern(entry)
        else:
            return jsonify({"error": f"no stop action for kind={kind}"}), 400
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": str(ex)}), 500
    return jsonify({"ok": True})


@app.route("/api/tools/<tool_id>/uninstall/preview", methods=["POST"])
def api_uninstall_preview(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    sizes = {}
    for p in entry.get("install_paths", []) + entry.get("config_paths", []) + ([entry["model_dir"]] if entry.get("model_dir") else []):
        full = expand(p)
        if Path(full).exists():
            r = subprocess.run(["du", "-sh", full], capture_output=True, text=True)
            sizes[p] = r.stdout.split()[0] if r.stdout else "?"
    return jsonify({"uninstall_cmds": entry.get("uninstall_cmds", []), "sizes": sizes})


@app.route("/api/tools/<tool_id>/uninstall/confirm", methods=["POST"])
def api_uninstall_confirm(tool_id):
    body = request.get_json(force=True, silent=True) or {}
    if body.get("confirm_step") not in (1, 2):
        return jsonify({"error": "confirm_step 1 then 2 required"}), 400
    if body.get("confirm_step") == 1:
        return jsonify({"ok": True, "next": "call again with confirm_step=2 to execute"})

    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404

    output = []
    for cmd in entry.get("uninstall_cmds", []):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path.home())
        output.append({"cmd": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})

    entries = [e for e in entries if e["id"] != tool_id]
    save_registry(entries)
    obsidian_sync.mark_uninstalled(tool_id)
    return jsonify({"ok": True, "output": output})


@app.route("/api/tools/<tool_id>/check_update")
def api_check_update(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    if not entry.get("installed_version_cmd") or not entry.get("latest_version_cmd"):
        return jsonify({"error": "no update check configured for this tool"}), 400
    installed = run_version_cmd(entry["installed_version_cmd"])
    latest = run_version_cmd(entry["latest_version_cmd"], timeout=30)
    return jsonify({
        "installed": installed,
        "latest": latest,
        "update_available": bool(installed and latest and installed != latest),
    })


@app.route("/api/tools/<tool_id>/update", methods=["POST"])
def api_update(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    cmd = entry.get("update_cmd")
    if not cmd:
        return jsonify({"error": "no update_cmd configured for this tool"}), 400
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900, cwd=Path.home())
    except subprocess.TimeoutExpired:
        return jsonify({"error": "update command timed out after 15 minutes"}), 500
    return jsonify({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "output": (r.stdout + r.stderr)[-4000:],
        "new_version": run_version_cmd(entry.get("installed_version_cmd")),
    })


@app.route("/api/tools/<tool_id>/launch")
def api_launch(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry or not entry.get("launch_url"):
        return jsonify({"error": "no launch url"}), 404
    return jsonify({"launch_url": entry["launch_url"]})


@app.route("/api/tools/<tool_id>/launch_terminal", methods=["POST"])
def api_launch_terminal(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry or not entry.get("launch_cmd"):
        return jsonify({"error": "no launch_cmd for this tool"}), 404
    # Keep the window open after the command exits (fast exit/error shouldn't
    # just vanish the terminal before you can read it). Passed as separate
    # argv elements (via "--") rather than one shell string, so nothing here
    # needs manual quote-escaping.
    #
    # x-terminal-emulator (the Debian alternatives wrapper) does NOT reliably
    # forward args past "--" on this system - verified by hand: the tab opens
    # but lands on a bare shell prompt, command never runs. Calling
    # gnome-terminal directly does work (verified: a `sleep 60` marker showed
    # up as a real child process). Prefer it when present, fall back to the
    # wrapper for other desktops - may have the same issue there, untested.
    terminal = shutil.which("gnome-terminal") or shutil.which("x-terminal-emulator")
    if not terminal:
        return jsonify({"error": "no terminal emulator found (looked for gnome-terminal, x-terminal-emulator)"}), 500
    inner = f"{entry['launch_cmd']}; echo; echo '[exited - press Enter to close]'; read"
    try:
        subprocess.Popen([terminal, "--", "bash", "-c", inner], cwd=Path.home(), start_new_session=True)
    except FileNotFoundError:
        return jsonify({"error": f"failed to launch {terminal}"}), 500
    return jsonify({"ok": True})


@app.route("/api/scan")
def api_scan():
    entries = load_registry()
    return jsonify(discover.find_candidates(entries))


@app.route("/api/registry/add", methods=["POST"])
def api_registry_add():
    body = request.get_json(force=True)
    entries = load_registry()
    if any(e["id"] == body.get("id") for e in entries):
        return jsonify({"error": "id already exists"}), 400
    entries.append(body)
    save_registry(entries)
    return jsonify({"ok": True})


@app.route("/api/sync", methods=["POST"])
def api_sync():
    entries = load_registry()
    for e in entries:
        e["status"] = check_status(e)
        e["models"] = get_models(e)
    obsidian_sync.sync_all(entries)
    return jsonify({"ok": True})


@app.route("/api/query", methods=["POST"])
def api_query():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    backend_id = body.get("backend", "auto")
    if not question:
        return jsonify({"error": "question required"}), 400
    entries = load_registry()
    for e in entries:
        e["status"] = check_status(e)
        e["models"] = get_models(e)
    result = query.ask(question, entries, backend_id)
    return jsonify(result)


@app.route("/api/backends")
def api_backends():
    return jsonify(query.list_backends_redacted())


@app.route("/api/backends/probe_models", methods=["POST"])
def api_probe_models():
    body = request.get_json(force=True)
    try:
        models = query.probe_models(
            body.get("id", ""), body.get("type", ""), body.get("base_url", ""), body.get("api_key", "")
        )
        return jsonify({"models": models})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        body = request.get_json(force=True)
        query.save_settings(body.get("backends", []), body.get("default_backend", "auto"), body.get("common_questions", []))
        return jsonify({"ok": True})
    return jsonify(query.list_backends_redacted())


if __name__ == "__main__":
    # Debug/reload is handy for interactive dev but shouldn't run unattended as
    # a boot-time service - opt in explicitly rather than defaulting it on.
    debug = os.environ.get("LLM_CHOOSER_DEBUG") == "1"
    app.run(host="127.0.0.1", port=7890, debug=debug)
