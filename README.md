# 🚩 Muster

A local dashboard for people who install too many LLM tools and lose track
of them. Muster keeps one registry of everything you've got running on your
machine — model runtimes, agent CLIs, image/video generators, whatever — and
gives you:

- **Live status** for every tool: running/stopped, port open/closed, models available, whether it's set to start at boot/login (systemd enabled state, or Docker restart policy)
- **Hardware bar** across the top — GPU/VRAM, RAM, CPU cores, auto-detected on whatever machine it's running on (see `hardware.py`)
- **Start / Stop / Launch / Uninstall** from one page, grouped into tabs by category
- **Update checks** — per-tool, on demand: compares the installed version against what's actually latest, with a one-click Update if you've wired up the command (see below)
- **A Models tab** — every downloaded model across all your tools, with size, path, and which tool(s) use it
- **A Projects tab** — group tools that belong to the same external project separately from the tool-type tabs
- **Discovery** — scan for installed things not yet in the registry (Docker containers, `~/.local/bin` executables), add them with one click
- **Obsidian sync** — pushes live status into your vault without ever touching your own handwritten notes
- **An AI query box** — ask plain-English questions about your setup, answered by a local model first, with your explicit confirmation before it ever falls back to a cloud API

It's a single Flask app with vanilla JS, no build step, no database — just a
JSON file describing your tools and a page that reads it.

## Setup

```
git clone <this repo> muster
cd muster
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp registry.example.json registry.json   # then edit it for your own tools, see below
venv/bin/python app.py
```

Open http://127.0.0.1:7890.

## The registry

`registry.json` (gitignored — it's your personal setup, not part of the repo)
is the source of truth for what tools exist and how to control them. See
`registry.example.json` for annotated examples of each `kind`:

| kind | for tools that... |
|---|---|
| `systemd-system` | run as a system-level `systemctl` service |
| `systemd-user` | run as a `systemctl --user` service |
| `docker-compose` | are one service in a `docker-compose.yml` |
| `docker-container` | are a plain Docker container |
| `process` | you start/stop as a bare process (needs `start_cmd`, and a `status_cmd` using **absolute paths** — relative paths make status detection unreliable once something else changes the working directory) |
| `llama-server` | a `llama-server` binary that takes a model path at start time instead of a fixed `start_cmd` — see below |
| `manual` | have no controllable running state (a CLI tool, a folder of weights) — status always shows `n/a` |

The "starts at boot/login" indicator is only shown for kinds with a real
mechanism to check: `systemctl is-enabled` for the two systemd kinds, and
the container's restart policy (`always`/`unless-stopped`) for the two
Docker kinds. `process`/`manual` entries have no generic equivalent, since
that would need a separate systemd unit of their own that isn't modeled here.

A `process` or `manual` entry can also set `launch_cmd` (e.g. `"lazydocker"`)
for tools that are only usable interactively in a terminal — no web UI to
open. This adds a **Launch in Terminal** button that opens a real terminal
window running that command, using `gnome-terminal` directly if present
(the generic `x-terminal-emulator` alternatives wrapper doesn't reliably
forward arguments past `--` on every system, so it's a fallback, not the
default). This is different from `launch_url`, which just opens a browser
tab — use whichever one actually matches how the tool is used; several CLI
tools (e.g. `dsh`, `hermes-agent`) turn out to have their own local web
dashboards worth wiring up as `process` entries with a real `launch_url`
instead of treating them as bare unlaunchable CLIs.

### The `llama-server` kind

Unlike the other kinds, `llama-server` has no fixed `start_cmd` — `llama.cpp`'s
server takes the model to load as a `-m` flag at process start, so Muster
needs to know *which* model before it can build the command. The card shows
a dropdown (populated the same way as `model_scan: "hf-cache-gguf"` for any
other tool) instead of a read-only model list; picking one and clicking
**Start** builds `llama-server -m <path> --alias <repo:quant> --host --port`
plus GPU-offload flags from `hardware.py`'s detection of *this* machine
(`-ngl 999` if a GPU is present, `-ngl 0` otherwise, `--threads` = CPU core
count) — so the same registry entry works unmodified on a different box.
`--alias` keeps the model's reported name as `repo:quant` instead of the
resolved HF-cache blob path (which is an unreadable hash).

Starting while an instance is already running swaps models cleanly: Muster
stops the old process and waits for the port to free before launching the
new one, rather than requiring a manual Stop first (a second `llama-server`
can't bind the same port, so without this the old model would keep serving
silently on click).

Needs a `binary` field (path to the `llama-server` executable) instead of
`start_cmd`, plus the usual `port`/`host`/`status_cmd`. See
`registry.example.json` for a full annotated entry.

### Update checks

Three optional fields, independent of `kind` - add them to any entry once
you've hand-verified the commands actually work for that tool:

- `installed_version_cmd` - prints the currently installed version (last non-blank line of stdout is used)
- `latest_version_cmd` - prints the latest available version, the same way
- `update_cmd` - actually runs the update

**Check for updates** compares the two strings - any mismatch is shown as
"update available", so `latest_version_cmd` needs to report the same kind of
string `installed_version_cmd` does (a git short-hash vs. a semver string
will always look different). If both are set, the card gets a **Check for
updates** button; if an update's available and `update_cmd` is also set, an
**Update** button appears too, with a confirmation dialog showing exactly
what will run.

**A tool's CLI can change shape between versions** - this happened for real
during development: `unsloth`'s bare launch command started requiring an
explicit subcommand after an update, breaking its `start_cmd` silently until
someone clicked Start and got an error. Muster can't detect that kind of
breakage automatically (it would mean actually invoking the tool speculatively,
which isn't something to do without asking), so after running an update it's
worth clicking Start/Stop once to confirm they still work, and updating
`start_cmd`/`stop_cmd`/`status_cmd` in `registry.json` if not. Don't wire
`update_cmd` up for a tool whose update behavior you haven't checked by hand
first - it runs verbatim, with no extra confirmation beyond the one dialog.

Add entries by hand, or click **Scan for new tools** in the UI to find
candidates (Docker containers and executables in `~/.local/bin` not already
tracked) and add them with one click — nothing is ever added automatically.

A `"project": "some-name"` field on any entry pulls it out of its category
tab and into the **Projects** tab instead, grouped by project name — useful
when several tools belong to one external thing (e.g. a docker-compose stack
for a specific app) rather than being general-purpose tools in their own right.

## The AI query box

Backends are configured in **Settings** (⚙ button) — each one is either:
- `openai` — any OpenAI-compatible chat endpoint (a local `llama-server`, Ollama's `/v1` API, Unsloth Studio, LM Studio, etc.)
- `anthropic` — Claude

Mark a backend `external: true` if it leaves your machine. In **Auto** mode,
Muster tries your local (`external: false`) backends first; if none of them
answer, it does **not** silently fall back to a cloud API — it asks you to
confirm first, since that means sending your question (plus your tool/vault
context) to an external service. Pick a specific backend from the dropdown
to skip that gate entirely.

Each backend row has a **⟳ Load models** button that queries the backend
live for what's actually available and turns the model field into a dropdown.

The AI is explicitly told it's **read-only** — it can't start/stop/uninstall
anything, only report on the state it's given. If you ask it to perform an
action, it's instructed to say so and point you at the real button instead
of claiming to have done it.

Config lives in `config.local.json` (gitignored, holds real API keys):

```json
{
  "backends": [
    {
      "id": "ollama", "label": "Ollama (local)", "type": "openai",
      "base_url": "http://127.0.0.1:11434/v1", "api_key": "", "model": "qwen3:8b",
      "external": false
    },
    {
      "id": "claude", "label": "Claude (cloud)", "type": "anthropic",
      "api_key": "sk-ant-...", "model": "claude-haiku-4-5",
      "external": true
    }
  ],
  "default_backend": "auto",
  "common_questions": ["What's running right now?"]
}
```

## Obsidian sync

Click **Sync to Obsidian** to push live status into a vault. Set the vault
path in `obsidian_sync.py` (`VAULT`/`TOOLS_DIR` constants). Each tool gets a
note with a machine-managed block between
`<!-- llm-chooser:managed:start/end -->` markers (status, port, models,
install paths); **everything else in the note is left completely
untouched** — your own handwritten prose survives every sync. Uninstalling a
tool doesn't delete its note, just appends an "Uninstalled" marker.

Sync is manual (button-triggered), not automatic on every status poll.

## Running on boot

A sample systemd user unit:

```ini
[Unit]
Description=Muster - local LLM tool dashboard
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/path/to/muster
ExecStart=/path/to/muster/venv/bin/python /path/to/muster/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

```
systemctl --user enable --now muster.service
loginctl enable-linger $USER   # so it starts at boot without needing a login session
```

The dev server enables debug/auto-reload only when `LLM_CHOOSER_DEBUG=1` is
set — leave it unset for a boot-time service.

## Limitations

- No auth on the dashboard — it binds to `127.0.0.1` only, treat it like any
  other localhost dev tool. Don't expose it beyond your own machine.
- Uses Flask's built-in dev server. Fine for a single-user localhost tool;
  not meant for anything beyond that.
- The query box's context is the whole registry + a handful of Obsidian
  notes stuffed into one prompt — no embeddings/retrieval, which is fine at
  the scale of "your own installed tools" but won't scale to a huge vault.
