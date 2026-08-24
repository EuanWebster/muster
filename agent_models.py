"""Read the configured model(s) for agent CLIs that don't expose a live API to query.

Each of these tools writes its own config file with a different shape - one
small function per tool rather than a generic scanner, since there's no
shared schema worth abstracting over for three formats.
"""
import json
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def dsh_models() -> list[str]:
    data = _load_yaml("~/.dsh/settings.yaml")
    default = data.get("agent-default-model") or {}
    default_id = f"{default.get('provider')}:{default.get('model')}" if default.get("model") else None

    others = []
    for provider, cfg in (data.get("llm-pi-ai", {}).get("providers") or {}).items():
        for m in cfg.get("models", []):
            mid = m.get("id") if isinstance(m, dict) else m
            others.append(f"{provider}:{mid}")

    result = [f"{default_id} (default)"] if default_id else []
    result += [m for m in others if m != default_id]
    return result


def hermes_models() -> list[str]:
    data = _load_yaml("~/.hermes/config.yaml")
    default = (data.get("model") or {}).get("default")

    others = []
    for cfg in (data.get("providers") or {}).values():
        others.extend(cfg.get("models", []))

    result = [f"{default} (default)"] if default else []
    result += [m for m in others if m != default]
    return result


def ollama_integration_models(integration_key: str) -> list[str]:
    p = Path("~/.ollama/config.json").expanduser()
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get("integrations", {}).get(integration_key, {}).get("models", [])
