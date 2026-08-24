"""Sync registry state into the Obsidian vault without clobbering handwritten notes.

Each tool's note gets a machine-owned block between MARK_START/MARK_END. Everything
outside that block (frontmatter aside) is left exactly as the user wrote it. On first
sync, notes have no markers yet - the block is inserted after frontmatter rather than
overwriting anything.
"""
import re
from datetime import date
from pathlib import Path

VAULT = Path.home() / "Documents/Obsidian Vault/Linux LLM Information"
TOOLS_DIR = VAULT / "Installed LLM Tools"

# Kept as "llm-chooser" (the project's old name) rather than renamed to "muster" -
# these strings are already embedded in real synced vault notes on disk, and
# changing them would break matching against existing notes (they'd look
# marker-less and get a duplicate block inserted instead of updated in place).
MARK_START = "<!-- llm-chooser:managed:start -->"
MARK_END = "<!-- llm-chooser:managed:end -->"

# Map registry ids to existing note filenames where they differ from f"{id}.md"
NOTE_NAME_OVERRIDES = {
    "dsh": "DeepSeek Harness (DSH).md",
    "openclaw": "GitHub Openclaw Access.md",
    "hermes-agent": "Hermes Agent.md",
}


def _note_path(tool_id: str) -> Path:
    name = NOTE_NAME_OVERRIDES.get(tool_id, f"{tool_id}.md")
    return TOOLS_DIR / name


def _managed_block(entry: dict) -> str:
    models = entry.get("models") or []
    model_lines = "\n".join(f"- `{m}`" for m in models) if models else "_none found_"
    paths = ", ".join(entry.get("install_paths", [])) or "_n/a_"
    config = ", ".join(entry.get("config_paths", [])) or "_n/a_"
    return "\n".join([
        MARK_START,
        "## Status",
        f"- Running: **{entry.get('status', 'unknown')}** (synced {date.today().isoformat()})",
        f"- Port: {entry.get('port') or '_n/a_'} ({entry.get('host', '')})",
        f"- Launch: {entry.get('launch_url') or '_n/a_'}",
        "",
        "## Models",
        model_lines,
        "",
        "## Install",
        f"- Paths: {paths}",
        f"- Config: {config}",
        MARK_END,
    ])


def _frontmatter(entry: dict) -> str:
    return "\n".join([
        "---",
        f"tags: [llm-tool, {entry.get('category', 'uncategorized')}]",
        f"status: {entry.get('status', 'unknown')}",
        f"port: {entry.get('port')}",
        f"updated: {date.today().isoformat()}",
        "---",
    ])


def sync_tool(entry: dict) -> None:
    path = _note_path(entry["id"])
    block = _managed_block(entry)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{_frontmatter(entry)}\n\n{block}\n\n## Notes\n")
        return

    text = path.read_text()

    if MARK_START in text and MARK_END in text:
        new_text = re.sub(
            re.escape(MARK_START) + r".*?" + re.escape(MARK_END),
            block,
            text,
            flags=re.DOTALL,
        )
        path.write_text(new_text)
        return

    # No markers yet: insert after frontmatter (or at the top), touch nothing else.
    fm_match = re.match(r"^---\n.*?\n---\n", text, flags=re.DOTALL)
    if fm_match:
        insert_at = fm_match.end()
        new_text = text[:insert_at] + f"\n{block}\n" + text[insert_at:]
    else:
        new_text = f"{block}\n\n" + text
    path.write_text(new_text)


def mark_uninstalled(tool_id: str) -> None:
    path = _note_path(tool_id)
    if not path.exists():
        return
    with path.open("a") as f:
        f.write(f"\n\n---\n_Uninstalled via Muster on {date.today().isoformat()}._\n")


def sync_all(entries: list[dict]) -> None:
    for entry in entries:
        sync_tool(entry)
