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
    # Same kinds check_boot_enabled can report on - the ones with a real
    # boot mechanism to toggle. process/llama-server can start and stop but
    # have nothing generic to enable, so they get no checkbox.
    can_boot = kind in ("systemd-system", "systemd-user", "docker-compose", "docker-container")
    return {"can_start": can_start, "can_stop": can_stop, "can_boot": can_boot}


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


LOG_DIR = Path.home() / ".cache" / "muster" / "logs"


def spawn_logged(cmd, name: str, shell: bool = False) -> str | None:
    """Start a background process, logging to a file rather than a pipe.

    stdout MUST NOT be subprocess.PIPE: nothing drains it after the liveness
    check below, so the child deadlocks the moment it fills the 64K kernel pipe
    buffer (Unsloth Studio managed that in ~7 minutes of logging, taking its
    HTTP loop down with it). A file never blocks the writer.

    Most launch failures (missing deps, wrong path, bad venv) surface within a
    couple seconds - catch those and report them instead of failing silently.
    A real server that's still booting after this is assumed to be fine.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, shell=shell, cwd=Path.home(), stdout=log,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
    time.sleep(2)
    if proc.poll() is not None:
        return (f"Process exited immediately (code {proc.returncode}):\n"
                + log_path.read_text(errors="replace")[-2000:])
    return None


def start_entry(entry: dict, model_key: str | None = None) -> str | None:
    """Starts one registry entry. Returns an error message, or None on success.

    Split out of api_start so an agent entry can bring up its inference engine
    through exactly the same dispatch instead of a second copy of it.
    """
    kind = entry["kind"]
    body = {"model": model_key}
    try:
        if kind == "systemd-system":
            subprocess.run(["sudo", "systemctl", "start", entry["unit"]], check=True, timeout=15,
                            capture_output=True, text=True)
        elif kind == "systemd-user":
            subprocess.run(["systemctl", "--user", "start", entry["unit"]], check=True, timeout=15,
                            capture_output=True, text=True)
        elif kind == "docker-compose":
            subprocess.run(["docker", "compose", "-f", expand(entry["compose_file"]), "up", "-d", entry["service"]],
                            check=True, timeout=60, capture_output=True, text=True)
        elif kind == "docker-container":
            subprocess.run(["docker", "start", entry["container"]], check=True, timeout=30,
                            capture_output=True, text=True)
        elif kind == "process" and entry.get("start_cmd"):
            err = spawn_logged(entry["start_cmd"], entry["id"], shell=True)
            if err:
                return err
        elif kind == "llama-server":
            model_key = body.get("model")
            if not model_key:
                return "pick a model first"
            record = next((r for r in hf_scan.scan_gguf_detailed() if r["key"] == model_key), None)
            if not record:
                return f"model not found in HF cache: {model_key}"
            binary = expand(entry["binary"])
            if not Path(binary).exists():
                return f"llama-server binary not found: {binary}"
            # Switching models: a second llama-server can't bind the same port, so
            # replace the running one instead of leaving it serving the old model.
            if check_status(entry) == "running":
                stop_by_pattern(entry)
                for _ in range(20):
                    if not port_open(entry.get("host") or "127.0.0.1", entry["port"]):
                        break
                    time.sleep(0.25)
            # --jinja: use the model's own chat template, so a reasoning model's
            # thinking comes back in reasoning_content instead of inline in content,
            # and per-request chat_template_kwargs (enable_thinking, reasoning_effort)
            # actually reach the template.
            cmd = [binary, "-m", record["path"], "--alias", model_key, "--host", entry.get("host", "127.0.0.1"),
                   "--port", str(entry["port"]), "--jinja"] + hardware.llama_server_flags(hardware.detect())
            # Per-model flags: speculative decoding is architecture-specific (draft-mtp
            # only loads on models with MTP heads), so it can't be a blanket flag or every
            # non-MTP model in the dropdown fails to load. "default" applies to the rest.
            model_args = entry.get("model_args") or {}
            cmd += [str(a) for a in model_args.get(model_key, model_args.get("default", []))]
            err = spawn_logged(cmd, entry["id"])
            if err:
                return err
        else:
            return f"no start action for kind={kind}"
    except subprocess.CalledProcessError as ex:
        return (ex.stderr or ex.stdout or str(ex)).strip()
    return None


@app.route("/api/tools/<tool_id>/start", methods=["POST"])
def api_start(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    body = request.get_json(silent=True) or {}
    err = prepare_engine(entry, body, entries) or start_entry(entry, body.get("model"))
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"ok": True})


def prepare_engine(agent: dict, body: dict, entries: list[dict]) -> str | None:
    """For an agent entry declaring `engines`: bring up the chosen inference
    engine on the chosen model and point the agent's own config at it.

    Returns an error message, or None (including when no engine was chosen, so
    an agent can still be started the plain way).
    """
    engine_id = body.get("engine")
    if not engine_id:
        return None
    if engine_id not in (agent.get("engines") or []):
        return f"{agent['id']} is not configured to use engine '{engine_id}'"
    engine = next((e for e in entries if e["id"] == engine_id), None)
    if not engine:
        return f"engine not in registry: {engine_id}"
    model_key = body.get("model")
    if not model_key:
        return "pick a model first"

    # llama-server takes its model at launch, so it always gets (re)started to
    # guarantee the chosen one is what's actually serving. Studio loads models
    # on demand, so leave it alone if it's already up.
    if engine["kind"] == "llama-server" or check_status(engine) != "running":
        err = start_entry(engine, model_key)
        if err:
            return f"failed to start {engine_id}: {err}"

    host, port = engine.get("host") or "127.0.0.1", engine.get("port")
    if port:
        # llama-server only binds the port once the model is loaded - a big
        # quant off a cold cache is minutes, not seconds.
        for _ in range(360):
            if port_open(host, port):
                break
            time.sleep(0.5)
        else:
            return f"{engine_id} did not open port {port} in time"

    # Studio advertises a repo id and picks the quant itself; llama-server is
    # launched with --alias <repo:quant>, so there the full key is the model id.
    served = model_key if engine["kind"] == "llama-server" else model_key.split(":")[0]
    return agent_models.set_backend(agent["id"], f"http://{host}:{port}/v1", served)


@app.route("/api/tools/<tool_id>/stop", methods=["POST"])
def api_stop(tool_id):
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    kind = entry["kind"]
    try:
        if kind == "systemd-system":
            subprocess.run(["sudo", "systemctl", "stop", entry["unit"]], check=True, timeout=15,
                            capture_output=True, text=True)
        elif kind == "systemd-user":
            subprocess.run(["systemctl", "--user", "stop", entry["unit"]], check=True, timeout=15,
                            capture_output=True, text=True)
        elif kind == "docker-compose" or kind == "docker-container":
            subprocess.run(["docker", "stop", entry["container"]], check=True, timeout=30,
                            capture_output=True, text=True)
        elif kind in ("process", "llama-server") and entry.get("stop_cmd"):
            subprocess.run(shlex.split(entry["stop_cmd"]), check=True, timeout=15,
                            capture_output=True, text=True)
        elif kind in ("process", "llama-server") and entry.get("status_cmd"):
            stop_by_pattern(entry)
        else:
            return jsonify({"error": f"no stop action for kind={kind}"}), 400
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": (ex.stderr or ex.stdout or str(ex)).strip()}), 500
    return jsonify({"ok": True})


@app.route("/api/tools/<tool_id>/boot", methods=["POST"])
def api_boot(tool_id):
    """Enable/disable start-at-boot for the kinds that have a real mechanism.

    Deliberately does not touch the running state: enabling a stopped service
    doesn't start it, disabling a running one doesn't stop it - same split
    systemctl itself uses between enable and start.
    """
    entries = load_registry()
    entry = next((e for e in entries if e["id"] == tool_id), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    kind = entry["kind"]
    enabled = bool((request.get_json(silent=True) or {}).get("enabled"))
    try:
        if kind == "systemd-system":
            subprocess.run(["sudo", "systemctl", "enable" if enabled else "disable", entry["unit"]],
                            check=True, timeout=15, capture_output=True, text=True)
        elif kind == "systemd-user":
            subprocess.run(["systemctl", "--user", "enable" if enabled else "disable", entry["unit"]],
                            check=True, timeout=15, capture_output=True, text=True)
        elif kind in ("docker-compose", "docker-container"):
            policy = "unless-stopped" if enabled else "no"
            subprocess.run(["docker", "update", f"--restart={policy}", entry["container"]],
                            check=True, timeout=30, capture_output=True, text=True)
        else:
            return jsonify({"error": f"no boot toggle for kind={kind}"}), 400
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": (ex.stderr or ex.stdout or str(ex)).strip()}), 500
    return jsonify({"ok": True, "boot_enabled": check_boot_enabled(entry)})


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
    # A terminal-launched agent picks its engine the same way a started one does.
    err = prepare_engine(entry, request.get_json(silent=True) or {}, entries)
    if err:
        return jsonify({"error": err}), 500
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
