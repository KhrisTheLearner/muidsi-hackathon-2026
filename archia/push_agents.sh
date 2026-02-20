#!/usr/bin/env bash
# push_agents.sh — Sync all AgriFlow agent configs + prompts to Archia cloud.
#
# Usage:
#   export ARCHIA_TOKEN=your_token
#   bash push_agents.sh [--dry-run]
#
# What it does:
#   For each enabled agent in archia/agents/*.toml, reads the prompt file,
#   parses [mcp_tools] sections, then PUTs to the Archia API.
#   Falls back to POST if the agent does not yet exist (404).
#
# Requires: python3 (stdlib only), bash >= 4

set -euo pipefail

# ── Token guard ────────────────────────────────────────────────────────────
: "${ARCHIA_TOKEN:?'ERROR: ARCHIA_TOKEN is not set.  Run: export ARCHIA_TOKEN=your_token'}"

# ── Args ───────────────────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Delegate to inline Python (handles TOML parsing + HTTP) ────────────────
python3 - "$SCRIPT_DIR" "$DRY_RUN" "$ARCHIA_TOKEN" <<'PYTHON'
# -*- coding: utf-8 -*-
import sys
import json
import re
import pathlib
from urllib import request, error

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

script_dir = pathlib.Path(sys.argv[1])
dry_run    = sys.argv[2] == "true"
token      = sys.argv[3]

agents_dir  = script_dir / "archia" / "agents"
prompts_dir = script_dir / "archia" / "prompts"
base_url    = "https://registry.archia.app"

HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}


# ── TOML parser: handles top-level scalars + [mcp_tools] section ───────────
def parse_toml(text: str) -> dict:
    """Parse agent TOML files.

    Supports:
      - key = "string"
      - key = true/false
      - [mcp_tools] section with:
          tool-name = null          (grant all tools)
          tool-name = ["a", "b"]    (grant specific tools)
    Keys may contain hyphens (e.g. agriflow-sqlite).
    Inline comments (# ...) are stripped on every line.
    """
    result: dict = {"mcp_tools": {}}
    current_section = None

    for raw_line in text.splitlines():
        # Strip inline comments, then whitespace
        line = raw_line.split("#")[0].strip()
        if not line:
            continue

        # Section header: [section-name]
        m = re.match(r'^\[([^\]]+)\]$', line)
        if m:
            current_section = m.group(1).strip()
            continue

        # Key = value  (keys may include hyphens)
        m = re.match(r'^([\w\-]+)\s*=\s*(.+)', line)
        if not m:
            continue
        key   = m.group(1)
        value = m.group(2).strip()

        if current_section == "mcp_tools":
            # Parse null or array of strings
            if value == "null":
                result["mcp_tools"][key] = None
            elif value.startswith("["):
                # Extract quoted strings from array literal
                items = re.findall(r'"([^"]*)"', value)
                result["mcp_tools"][key] = items if items else []
            else:
                result["mcp_tools"][key] = None  # treat unknown as null (all tools)
        else:
            # Top-level scalar values
            if value.startswith('"') and value.endswith('"'):
                result[key] = value[1:-1]
            elif value == "true":
                result[key] = True
            elif value == "false":
                result[key] = False

    return result


# ── HTTP helpers ────────────────────────────────────────────────────────────
def _call(method: str, url: str, payload: dict) -> tuple:
    data = json.dumps(payload).encode()
    req  = request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except error.HTTPError as exc:
        return exc.code, exc.read().decode()


def put_agent(name: str, payload: dict) -> tuple:
    return _call("PUT", f"{base_url}/v1/agent/config/{name}", payload)


def post_agent(payload: dict) -> tuple:
    return _call("POST", f"{base_url}/v1/agent/config", payload)


# ── Main sync loop ──────────────────────────────────────────────────────────
print(f"AgriFlow Agent Sync -> {base_url}")
if dry_run:
    print("  (DRY RUN — no API calls will be made)")
print("=" * 56)

ok = failed = skipped = 0

for toml_path in sorted(agents_dir.glob("*.toml")):
    cfg = parse_toml(toml_path.read_text(encoding="utf-8"))

    name    = cfg.get("name", "")
    enabled = cfg.get("enabled", True)

    if not name:
        print(f"  WARN  {toml_path.name} — missing 'name' field, skipping")
        skipped += 1
        continue

    if not enabled:
        print(f"  SKIP  {name} (disabled)")
        skipped += 1
        continue

    prompt_file = cfg.get("system_prompt_file", "")
    if not prompt_file:
        print(f"  WARN  {name} — missing 'system_prompt_file', skipping")
        skipped += 1
        continue

    prompt_path = prompts_dir / prompt_file
    if not prompt_path.exists():
        print(f"  ERROR {name} — prompt file not found: {prompt_file}")
        failed += 1
        continue

    system_prompt = prompt_path.read_text(encoding="utf-8")
    mcp_tools     = cfg.get("mcp_tools", {})

    payload = {
        "name":          name,
        "model_name":    cfg.get("model_name", "priv-claude-sonnet-4-5-20250929"),
        "enabled":       True,
        "description":   cfg.get("description", ""),
        "system_prompt": system_prompt,
    }

    # Include mcp_tools if any are configured
    if mcp_tools:
        payload["mcp_tools"] = mcp_tools

    # Include can_manage_agents if set
    if cfg.get("can_manage_agents") is not None:
        payload["can_manage_agents"] = cfg["can_manage_agents"]

    if dry_run:
        tool_summary = ", ".join(mcp_tools.keys()) if mcp_tools else "none"
        print(f"  DRY   {name}")
        print(f"         prompt: {len(system_prompt)} chars from {prompt_file}")
        print(f"         mcp_tools: [{tool_summary}]")
        ok += 1
        continue

    # Try PUT first (update existing agent)
    status, body = put_agent(name, payload)

    if status in (200, 201):
        tool_summary = ", ".join(mcp_tools.keys()) if mcp_tools else "none"
        print(f"  OK    {name}  (PUT {status})  mcp_tools=[{tool_summary}]")
        ok += 1

    elif status == 404:
        # Agent not found — create it with POST
        status, body = post_agent(payload)
        if status in (200, 201):
            print(f"  OK    {name}  (POST {status} — created)")
            ok += 1
        else:
            snippet = body[:120].replace("\n", " ")
            print(f"  FAIL  {name}  (POST {status}): {snippet}")
            failed += 1

    elif status == 409:
        # Conflict (duplicate) — retry with PUT
        status, body = put_agent(name, payload)
        if status in (200, 201):
            print(f"  OK    {name}  (PUT retry {status})")
            ok += 1
        else:
            snippet = body[:120].replace("\n", " ")
            print(f"  FAIL  {name}  (PUT retry {status}): {snippet}")
            failed += 1

    else:
        snippet = body[:120].replace("\n", " ")
        print(f"  FAIL  {name}  (PUT {status}): {snippet}")
        failed += 1

# ── Summary ─────────────────────────────────────────────────────────────────
print("=" * 56)
print(f"Results: {ok} OK,  {skipped} skipped,  {failed} failed")

if failed > 0:
    print(f"WARNING: {failed} agent(s) failed to sync.")
    sys.exit(1)
else:
    print("All enabled agents synced successfully.")
PYTHON
