"""Answer natural-language questions about installed tools + the Obsidian vault.

Backends are user-configured in config.local.json as a list, each either
"openai" (any OpenAI-compatible chat endpoint - Unsloth Studio, another local
llama-server, etc) or "anthropic" (Claude). "auto" mode tries local backends
in order; if none answer, it does NOT silently call an external backend - it
reports back so the caller can ask the user for confirmation first.
"""
import json
import os
from pathlib import Path

import requests

from obsidian_sync import VAULT, TOOLS_DIR

CONFIG_FILE = Path(__file__).parent / "config.local.json"

# Measured against this vault's actual JSON+markdown content: ~2.45 chars/token,
# denser than the "~4 chars/token" rule of thumb for prose.
UNSLOTH_CHAR_BUDGET = 12000 * 2.4

# The loaded model is a reasoning model that spends tokens on hidden
# reasoning_content before the visible answer (see ~/Projects/scutwork/CLAUDE.md's
# documented quirks) - a low max_tokens can burn the whole budget on reasoning
# and return empty content even on a "successful" call. Generation on local
# hardware is also slow for a big prompt (~30s+ observed), so the timeout needs
# real headroom too.
OPENAI_MAX_TOKENS = 2500
OPENAI_TIMEOUT = 120

DEFAULT_BACKENDS = [
    {
        "id": "unsloth",
        "label": "Unsloth Studio (local)",
        "type": "openai",
        "base_url": "http://127.0.0.1:8889/v1",
        "api_key": "",
        "model": "local",
        "external": False,
    },
    {
        "id": "claude",
        "label": "Claude (cloud)",
        "type": "anthropic",
        "api_key": "",
        "model": "claude-haiku-4-5",
        "external": True,
    },
]


DEFAULT_COMMON_QUESTIONS = [
    "What's running right now?",
    "What tools depend on other tools being up?",
    "What models are downloaded and where?",
]
MAX_COMMON_QUESTIONS = 10


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"backends": DEFAULT_BACKENDS, "default_backend": "auto", "common_questions": DEFAULT_COMMON_QUESTIONS}
    data = json.loads(CONFIG_FILE.read_text())
    if "backends" not in data:
        # migrate the old flat unsloth_url/unsloth_api_key/claude_api_key shape
        data["backends"] = [
            {**DEFAULT_BACKENDS[0], "base_url": data.get("unsloth_url", DEFAULT_BACKENDS[0]["base_url"]),
             "api_key": data.get("unsloth_api_key", "")},
            {**DEFAULT_BACKENDS[1], "api_key": data.get("claude_api_key", "")},
        ]
        data.setdefault("default_backend", "auto")
    data.setdefault("common_questions", DEFAULT_COMMON_QUESTIONS)
    return data


def save_config(data: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def probe_models(backend_id: str, backend_type: str, base_url: str, api_key: str) -> list[str]:
    """List models actually available from a backend, for the settings dropdown.

    A blank api_key means "use whatever's already saved for this backend id"
    - the settings form never round-trips real keys through the browser.
    """
    if not api_key and backend_id:
        existing = next((b for b in load_config()["backends"] if b["id"] == backend_id), None)
        if existing:
            api_key = existing.get("api_key", "")

    if backend_type == "openai":
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = requests.get(f"{base_url}/models", headers=headers, timeout=5)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json()["data"])

    if backend_type == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        return [m.id for m in client.models.list(limit=100).data]

    return []


def list_backends_redacted() -> dict:
    """For the settings UI - never send real api_key values back to the browser."""
    config = load_config()
    backends = [{**b, "api_key": bool(b.get("api_key")) and "configured" or ""} for b in config["backends"]]
    return {
        "backends": backends,
        "default_backend": config.get("default_backend", "auto"),
        "common_questions": config.get("common_questions", DEFAULT_COMMON_QUESTIONS),
    }


def save_settings(new_backends: list[dict], default_backend: str, common_questions: list[str]) -> None:
    """Merge with existing config: a blank api_key in an incoming backend keeps
    whatever key that backend id already had, so the settings form never has to
    round-trip real secrets through the browser."""
    existing = {b["id"]: b for b in load_config()["backends"]}
    merged = []
    for b in new_backends:
        prev = existing.get(b["id"], {})
        if not b.get("api_key"):
            b["api_key"] = prev.get("api_key", "")
        merged.append(b)
    common_questions = [q.strip() for q in common_questions if q.strip()][:MAX_COMMON_QUESTIONS]
    save_config({"backends": merged, "default_backend": default_backend, "common_questions": common_questions})


def _vault_files() -> list[tuple[str, str]]:
    """(title, content) pairs for Installed LLM Tools/*.md + Model Storage Map.md.

    Deliberately excludes any other vault note - two notes elsewhere hold a
    live API key and a GitHub PAT and must never enter a prompt.
    """
    files = []
    if TOOLS_DIR.exists():
        for md in sorted(TOOLS_DIR.glob("*.md")):
            files.append((md.stem, md.read_text()))
    storage_map = VAULT / "Model Storage Map.md"
    if storage_map.exists():
        files.append(("Model Storage Map", storage_map.read_text()))
    return files


def gather_context(entries: list[dict], vault_char_budget: int | None = None) -> str:
    registry_summary = json.dumps(
        [{"id": e["id"], "status": e.get("status"), "port": e.get("port"), "models": e.get("models")} for e in entries],
        indent=2,
    )
    files = _vault_files()
    per_file_budget = (vault_char_budget // max(len(files), 1)) if vault_char_budget else None
    parts = []
    for title, content in files:
        if per_file_budget and len(content) > per_file_budget:
            content = content[:per_file_budget] + "\n...[truncated]"
        parts.append(f"### {title}\n{content}")
    return f"## Live tool registry\n{registry_summary}\n\n## Obsidian notes\n" + "\n\n".join(parts)


def _build_prompt(question: str, entries: list[dict], vault_char_budget: int | None = None) -> str:
    context = gather_context(entries, vault_char_budget)
    return (
        "You answer questions about a set of locally-installed LLM tools, using the "
        "context below. You are read-only: you cannot start, stop, uninstall, or "
        "otherwise change anything on this system - you can only report on the state "
        "given to you. If asked to perform an action, say you can't and point to the "
        "relevant dashboard button/tab instead of claiming to have done it.\n\n"
        f"{context}\n\nQuestion: {question}"
    )


def _call_openai(backend: dict, prompt: str) -> str | None:
    headers = {"Authorization": f"Bearer {backend['api_key']}"} if backend.get("api_key") else {}
    base_url = backend["base_url"]
    try:
        health = requests.get(f"{base_url}/models", headers=headers, timeout=2)
        if health.status_code != 200:
            return None
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={"model": backend.get("model", "local"), "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": OPENAI_MAX_TOKENS},
            timeout=OPENAI_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or None
    except (requests.RequestException, KeyError, IndexError):
        return None


def _call_anthropic(backend: dict, prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or backend.get("api_key")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=backend.get("model", "claude-haiku-4-5"),
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return None


def _call_backend(backend: dict, question: str, entries: list[dict]) -> str | None:
    if backend["type"] == "openai":
        full_prompt = _build_prompt(question, entries)
        if len(full_prompt) <= UNSLOTH_CHAR_BUDGET:
            prompt = full_prompt
        else:
            overhead = len(full_prompt) - len(gather_context(entries))
            prompt = _build_prompt(question, entries, vault_char_budget=max(UNSLOTH_CHAR_BUDGET - overhead, 500))
        return _call_openai(backend, prompt)
    if backend["type"] == "anthropic":
        prompt = _build_prompt(question, entries)
        return _call_anthropic(backend, prompt)
    return None


def ask(question: str, entries: list[dict], backend_id: str = "auto") -> dict:
    backends = {b["id"]: b for b in load_config()["backends"]}

    if backend_id != "auto":
        backend = backends.get(backend_id)
        if not backend:
            return {"answer": f"Unknown backend '{backend_id}'.", "source": "none"}
        answer = _call_backend(backend, question, entries)
        if answer:
            return {"answer": answer, "source": backend_id}
        return {"answer": f"{backend['label']} didn't respond (not running, no key configured, or timed out).", "source": "none"}

    # auto mode: try local (non-external) backends in configured order first.
    # Stop and ask before ever calling an external backend automatically.
    ordered = list(backends.values())
    for backend in ordered:
        if backend.get("external"):
            continue
        answer = _call_backend(backend, question, entries)
        if answer:
            return {"answer": answer, "source": backend["id"]}

    external = next((b for b in ordered if b.get("external") and b.get("api_key")), None)
    if external:
        return {
            "needs_confirmation": True,
            "fallback_backend": external["id"],
            "fallback_label": external["label"],
        }

    return {"answer": "No backend available - the local model isn't reachable and no external backend is configured (see Settings).", "source": "none"}
