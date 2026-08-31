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


def set_backend(agent_id: str, base_url: str, model_id: str) -> str | None:
    """Point one agent CLI at an OpenAI-compatible endpoint + model.

    Returns an error message, or None on success. One writer per tool, same as
    the readers above - the config formats have nothing worth abstracting over.
    Everything not related to the endpoint (thinking/compat settings, installed
    packages, UI state) is preserved as-is.
    """
    if agent_id == "dsh":
        p = Path("~/.dsh/settings.yaml").expanduser()
        data = _load_yaml(str(p))
        providers = data.setdefault("llm-pi-ai", {}).setdefault("providers", {})
        # Route key must be one pi-ai's catalog does NOT ship: under a catalog
        # key ("openai", "anthropic", ...) a `models` list only narrows that
        # catalog, so a local model id resolves to nothing and the picker comes
        # up empty. A hand-declared route needs `api` and `baseURL` explicitly.
        provider = providers.setdefault("muster", providers.pop("openai", {}))
        provider["baseURL"] = base_url
        provider["api"] = "openai-completions"
        provider.setdefault("apiKeyEnv", "OPENAI_API_KEY")
        models = provider.get("models") or [{}]
        models[0] = {**models[0], "id": model_id}
        provider["models"] = models
        default = data.setdefault("agent-default-model", {})
        default["provider"], default["model"] = "muster", model_id
        p.write_text(yaml.safe_dump(data, sort_keys=False))
        return None

    if agent_id == "pi":
        models_path = Path("~/.pi/agent/models.json").expanduser()
        settings_path = Path("~/.pi/agent/settings.json").expanduser()
        if not models_path.exists():
            return f"pi config not found: {models_path}"
        data = json.loads(models_path.read_text())
        providers = data.setdefault("providers", {})
        # Reuse whichever provider block exists as the template so the model's
        # reasoning/compat/context settings survive an endpoint change.
        template = next(iter(providers.values()), {})
        model_template = (template.get("models") or [{}])[0]
        providers["muster"] = {
            **template,
            "baseUrl": base_url,
            "api": "openai-completions",
            # llama-server takes no real auth. Always a literal placeholder here
            # (not templated) - an env-var reference like $UNSLOTH_API_KEY can
            # silently fail to resolve depending on how pi's shell was launched
            # (e.g. .bashrc's interactive-shell guard skips it in some paths),
            # which hides every model from `/model` with no clear error.
            "apiKey": "local-llama-cpp",
            "models": [{**model_template, "id": model_id}],
        }
        models_path.write_text(json.dumps(data, indent=2) + "\n")
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings["defaultProvider"], settings["defaultModel"] = "muster", model_id
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        return None

    return f"no backend writer for agent '{agent_id}'"


def ollama_integration_models(integration_key: str) -> list[str]:
    p = Path("~/.ollama/config.json").expanduser()
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get("integrations", {}).get(integration_key, {}).get("models", [])
