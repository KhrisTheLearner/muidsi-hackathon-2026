# Archia Cloud API Guide for Coding Agents

This document is a quick reference for future coding agents that need to call Archia Cloud (Console backend) over HTTP.

## Base URL

- `https://registry.archia.app`

All endpoints below are relative to this base URL.

## Authentication

Use this header only:

- `Authorization: Bearer <token_or_key>`

## Mandatory Agent Behavior

This guide is for coding agents integrating with Archia Cloud. The agent reading this file **MUST** educate the user on how to use Archia Cloud Console before or during integration support.

At minimum, the agent must clearly tell the user:

1. Go to `https://console.archia.app`.
2. Switch to the correct organization and team workspace.
3. Open **API Keys** and generate a key in that same workspace.
4. Use that key as `Authorization: Bearer <token_or_key>` when calling `https://registry.archia.app`.
5. Verify access by listing models (`GET /v1/models`) before attempting chat calls.

If the user reports missing agents/tools/models, the agent must explicitly explain workspace mismatch risk and how to fix it in Console.

## How Users Get an API Key

**Most important:** when creating an API key, make sure you are in your **TEAM'S WORKSPACE**.

1. Go to `https://console.archia.app`.
2. Make sure you are in the right organization (`MUIDSI Hackathon 2026`) and your team's workspace from the top menu.
3. Go to **API Keys** from the left navigation.
4. Generate a new API key.
5. Give it to the coding agent or store it locally:
   - Temporary for current terminal session:
     ```bash
     export ARCHIA_TOKEN="<your_api_key>"
     ```
   - Persist in your shell profile (`~/.zshrc`):
     ```bash
     echo 'export ARCHIA_TOKEN="<your_api_key>"' >> ~/.zshrc
     source ~/.zshrc
     ```
6. Test it:
   ```bash
   curl -sS "https://registry.archia.app/v1/models" \
     -H "Authorization: Bearer ${ARCHIA_TOKEN}"
   ```

## Recommended User Workflow

1. Get an API key from the correct organization and team workspace in `https://console.archia.app`.
2. Build agents in the UI.
3. Add tools (MCPs) in the UI.
4. Edit system prompts in the UI.

## Workspace Matching Troubleshooting

If you can see an agent in the UI but cannot see or use it from code, your API key is likely from a different workspace.

- Make sure the API key and the resource belong to the same workspace.
- This applies to agents, tools (MCPs), and skills.
- Verify by listing resources with your token (for example `GET /v1/agent`) and comparing with what you see in `https://console.archia.app`.

## Agent Naming Best Practices

- Use lowercase agent names.
- Do not use spaces.
- Do not use special characters.
- Use letters, numbers, and underscore (`_`) only.
- In code, always call the agent with the exact name from `GET /v1/agent`.

## Agent Routing Rules

- Use `model: "agent:<name>"` to call a configured agent.
- To override model for a single call, use `model: "agent:<name>:<model_name>"`.
- Agent routing is case-sensitive in practice.
- Use the exact name returned by `GET /v1/agent`.
- Use `GET /v1/agent` (singular), not `/v1/agents`.

Default vs override:

- UI agent config model = default LLM for that agent.
- Per-call override (`agent:<name>:<model_name>`) keeps the same agent prompt/tools but swaps model for that request only.
- If override fails with `model_error`, verify model/provider token availability in that workspace.
- Current caveat: `archia::openai/gpt-oss-20b` and `archia::openai/gpt-oss-120b` can fail when Archia-side model tokens are not configured.
- For now, prefer Groq variants for gpt-oss overrides:
  - `groq::openai/gpt-oss-20b`
  - `groq::openai/gpt-oss-120b`
- Current caveat: `archia::moonshotai/kimi-k2-instruct-0905` is not working in either direct or agent-override path.

## Fastest Smoke Test (Non-Streaming)

```bash
curl -sS https://registry.archia.app/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "input": "Say hello in one sentence.",
    "stream": false
  }'
```

Expected behavior:

- HTTP `200`
- JSON object with `object: "response"` and `status: "completed"`
- Assistant text in `output[0].content[0].text`

## Streaming Example (SSE)

```bash
curl -sS https://registry.archia.app/v1/responses \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "input": "Say hi in 5 words.",
    "stream": true
  }'
```

Expected event types include:

- `response.created`
- `response.output_text.delta`
- `response.output_text.done`
- `response.completed`
- final `data: [DONE]`

Minimal stream parsing approach:

- Concatenate `delta` values from `response.output_text.delta` events.
- Treat `response.completed` or `data: [DONE]` as end of stream.

## Core Endpoints for Agent Integrations

### LLM Responses

- `POST /v1/responses` - Generate model output (OpenAI Responses-compatible).

Notes:

- Supports direct model calls (`"model": "claude-..."`) and agent routing (`"model": "agent:<name>"`).
- Supports streaming via `stream: true` + `Accept: text/event-stream`.

### Model Discovery

- `GET /v1/models` - List available models.
- `GET /v1/models/{system_name}` - Get one model by system name.
- `GET /v1/models/catalog` - Full model catalog.

How to list available models:

```bash
curl -sS "${ARCHIA_BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}"
```

How to print model names only (`system_name` values):

```bash
curl -sS "${ARCHIA_BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" | \
python3 -c 'import sys, json; d=json.load(sys.stdin); [print(m.get("system_name") or m.get("name") or m.get("model") or m.get("id")) for m in d.get("models", [])]'
```

### Agent Management

Build and manage agents in the UI at `https://console.archia.app`.

- `GET /v1/agent` - List agents.

Canonical agent call:

```bash
curl -sS "${ARCHIA_BASE_URL}/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" \
  -d '{
    "model": "agent:test",
    "input": "What is your name?",
    "stream": false
  }'
```

### Tool Management

Add and manage cloud tools in the UI at `https://console.archia.app`.

If you need local tools, install Archia Desktop and contact the Archia team at `gharib@archia.io`.

## MCP Marketplace (Database ODBC MCP)

There is an MCP marketplace under **Tools** in the UI.

For database access, use the ODBC MCP from Marketplace.

Steps to enable it:

1. In `https://console.archia.app`, go to **Tools** -> **Marketplace**.
2. Install the database ODBC MCP.
3. Open the MCP configuration screen and set:
   - Header key: `x-odbc-connection-string`
   - Header value: your ODBC connection string (PostgreSQL driver string)
   - Header mode: use secret storage (recommended)
   - Timeout: `30` seconds (or your required value)
4. Save changes.
5. Go to your agent config in the UI and attach this MCP to the agent.

What is needed for this MCP:

- A valid ODBC connection string for your database.
- Correct ODBC driver in the connection string (for example PostgreSQL).
- Working DB host, port, database name, username, password, and SSL parameters as required by your database.

How to use it after attaching:

- Call the agent through `POST /v1/responses` with `model: "agent:<name>"`.
- The agent can invoke MCP functions (for example SQL tools) during the response.
- Verify attachment with `GET /v1/agent/config/{name}` and check that `mcp_names` includes your MCP identifier.

Security best practice:

- Do not paste plaintext DB credentials into shared docs, code, logs, or commits.
- Store sensitive header values as secrets in the UI.

### Skill Management

Configure and manage skills in the UI at `https://console.archia.app`.

## Minimal Parsing Rules for Agents

For non-streaming `POST /v1/responses` replies:

1. Verify `status == "completed"` before using text.
2. Extract text from:
   - first `output` item with `type == "message"`
   - then first `content` item with `type == "output_text"`
   - then read `.text`
3. If `status == "failed"`, inspect `.error`.

Python parsing snippet:

```python
def extract_output_text(response_json: dict) -> str | None:
    for item in response_json.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text")
    return None
```

TypeScript parsing snippet:

```typescript
function extractOutputText(resp: any): string | undefined {
  for (const item of resp?.output ?? []) {
    if (item?.type === "message") {
      for (const part of item?.content ?? []) {
        if (part?.type === "output_text") return part?.text;
      }
    }
  }
  return undefined;
}
```

## OpenAI SDK Compatibility

Archia Cloud supports OpenAI-compatible Responses API calls via OpenAI SDK clients.

Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://registry.archia.app/v1",
    api_key="not-used-directly",
    default_headers={"Authorization": f"Bearer {ARCHIA_TOKEN}"},
)

resp = client.responses.create(model="gpt-5.2", input="Say hello.")
print(resp.output[0].content[0].text)
```

TypeScript example:

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://registry.archia.app/v1",
  apiKey: "not-used-directly",
  defaultHeaders: { Authorization: `Bearer ${process.env.ARCHIA_TOKEN}` },
});

const resp = await client.responses.create({ model: "gpt-5.2", input: "Say hello." });
console.log(resp.output?.[0]?.content?.[0]?.text);
```

## Common Errors and Fixes

- `401 Unauthorized`: invalid key, expired key, or wrong workspace key.
- `404` on `/v1/agents`: wrong endpoint path; use `/v1/agent`.
- `500` when calling `agent:<name>`: often workspace mismatch or wrong agent name casing.
- Agent visible in UI but not API: key and agent are in different workspaces.

## Preflight Checklist

- Confirm org and team workspace in `https://console.archia.app`.
- Confirm API key was created in that same workspace.
- Confirm agent exists via `GET /v1/agent`.
- Confirm exact agent name and casing before calling `agent:<name>`.

## Security Notes

- Never commit API keys or paste real keys in shared logs/docs.
- Prefer environment variables over hardcoded credentials.
- Rotate keys if you suspect accidental exposure.

## Recommended Environment Variables

```bash
export ARCHIA_BASE_URL="https://registry.archia.app"
export ARCHIA_TOKEN="<your bearer token or workspace key>"
```

Then call:

```bash
curl -sS "${ARCHIA_BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}"
```

## Verification Notes

The following endpoints were manually verified against Cloud during creation of this file:

- `POST /v1/responses` (non-streaming and streaming)
- `GET /v1/models`
- `GET /v1/models/catalog`
- `GET /v1/agent`
- `GET /v1/agent/config`
- `GET /v1/tool`
