Agent Configuration Overview

Complete guide to defining and configuring AI agents in Archia.
Configuration Options

Archia supports two methods for configuring agents:

    Individual TOML Files: Store each agent as a separate .toml file in ~/.archia/agents/
    Server Configuration File: Define agents inline in the server’s config.toml (legacy method)

The recommended approach is using individual TOML files, which allows for easier management, version control, and dynamic updates.
File-Based Agent Configuration
Location

Agent configuration files are stored in the agents directory:

    Production: ~/.archia/agents/
    Development: ~/.archia_dev/agents/

Each agent is defined in its own TOML file named {agent-name}.toml.
Basic Agent Definition

Every agent requires at minimum a name and model:

# ~/.archia/agents/assistant.toml
name = "assistant"
model_name = "claude-haiku-4-5-20251001"
enabled = true

This creates a basic agent accessible at /v1/agent/assistant/chat.
Complete Agent Structure

Here’s a fully-configured agent with all available options:

# ~/.archia/agents/expert.toml

# Required fields
name = "expert"                           # Unique identifier (alphanumeric, hyphens, underscores)
model_name = "claude-sonnet-4-5-20250929"   # LLM model to use
enabled = true                            # Whether the agent is active

# Optional description
description = "Expert technical assistant with database access"

# System prompt (choose one method)
system_prompt = "You are an expert assistant"     # Inline text
# OR
system_prompt_file = "expert.md"                  # Reference to file in ~/.archia/prompts/

# MCP tool access (new format - recommended)
[mcp_tools]
database = ["query", "list_tables"]       # Specific tools from 'database' MCP
filesystem = []                           # All tools from 'filesystem' MCP (empty array = all)
search = null                             # All tools (null = all)

# Legacy MCP access (deprecated, use mcp_tools instead)
# mcp_names = ["database", "filesystem"]

# Agent management capabilities
can_manage_agents = false                 # If true, agent can spawn/manage other agents

# Extended thinking (for supported models)
enable_extended_thinking = false          # Enable extended thinking for complex tasks

Configuration Fields Reference
Field	Type	Required	Description
name	String	Yes	Unique identifier for the agent
model_name	String	Yes	LLM model identifier (e.g., “claude-sonnet-4-5-20250929”)
enabled	Boolean	Yes	Whether the agent is active
description	String	No	Human-readable description
system_prompt	String	No	Inline system prompt text
system_prompt_file	String	No	Reference to prompt file in ~/.archia/prompts/
mcp_tools	Map	No	MCP name → list of tool names (or null/empty for all)
mcp_names	Array	No	Legacy: List of MCP server names (deprecated)
can_manage_agents	Boolean	No	Allow agent to spawn/manage other agents
enable_extended_thinking	Boolean	No	Enable extended thinking/reasoning mode (see below)
System Prompts

System prompts define your agent’s personality, knowledge, and behavior. You can specify them in three ways:
1. Inline Prompts

For shorter prompts, include them directly in the configuration:

name = "comedian"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
system_prompt = """
You are a professional comedian who explains technical concepts through humor.
Use puns, jokes, and funny analogies to make complex topics accessible.
Always stay respectful and appropriate.
"""

2. File-based Prompts

For complex prompts, reference external files stored in ~/.archia/prompts/:

name = "analyst"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
system_prompt_file = "data_analyst.md"

Create the prompt file at ~/.archia/prompts/data_analyst.md:

# Data Analyst Assistant

You are an expert data analyst with 20 years of experience in:

- Statistical analysis and hypothesis testing
- Data visualization and storytelling
- SQL and database optimization
- Python (pandas, numpy, scikit-learn)
- Business intelligence and KPI development

## Communication Style

- Be precise with statistical terminology
- Provide confidence intervals when appropriate
- Suggest visualizations for insights
- Always consider data quality and biases

Prompt Resolution Priority

When both system_prompt and system_prompt_file are specified:

    system_prompt (inline) takes precedence
    system_prompt_file is used as fallback

Extended Thinking

The enable_extended_thinking option enables reasoning capabilities for supported models, allowing them to “think through” complex problems before responding.
Basic Usage

name = "analyst"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
enable_extended_thinking = true

system_prompt = "You are an expert data analyst."

Behavior by Model

When enable_extended_thinking = true:
Model Type	Behavior
All models	Reasoning effort set to High

When enable_extended_thinking = false (default):
Model Type	Behavior
GPT models (gpt-*, gpt-oss-*)	Reasoning effort set to Medium
Claude models	No extended thinking (disabled)
Other models	No extended thinking (disabled)
Provider-Specific Implementation

Different providers implement extended thinking differently:
Provider	Implementation
Anthropic	Enables “extended thinking” with a token budget (~24K tokens for High)
Google	Uses thinkingBudget parameter
OpenAI	Passes reasoning_effort keyword to the API
When to Use Extended Thinking

Enable for:

    Complex analytical tasks
    Multi-step reasoning problems
    Code review and debugging
    Mathematical or logical problems
    Research synthesis

Keep disabled for:

    Simple Q&A
    Quick lookups
    Conversational chat
    Tasks where latency matters

Example: Research Agent with Extended Thinking

name = "research-analyst"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
enable_extended_thinking = true
description = "Deep research analyst for complex analysis tasks"

system_prompt = """
You are a senior research analyst. Take your time to think through problems 
thoroughly before providing your analysis. Consider multiple perspectives 
and potential edge cases.
"""

[mcp_tools]
web_search = null
document_store = ["search", "retrieve"]

Overriding via API

When using the Responses API, you can override the agent’s enable_extended_thinking setting by explicitly providing the reasoning parameter:

{
  "model": "agent:research-analyst",
  "input": "Analyze this data",
  "reasoning": {
    "effort": "low"
  }
}

This explicit reasoning parameter takes precedence over the agent’s configuration.
MCP Tool Integration
New Format (Recommended)

The mcp_tools map provides fine-grained control over which tools from each MCP server the agent can access:

[mcp_tools]
# Grant access to specific tools only
database = ["query", "list_tables", "describe_table"]

# Grant access to all tools from an MCP (two equivalent ways)
filesystem = []      # Empty array = all tools
search = null        # null = all tools

# Another MCP with specific tools
github = ["list_repos", "create_issue"]

Legacy Format (Deprecated)

The mcp_names array grants access to all tools from listed MCPs:

mcp_names = ["database", "filesystem", "search"]

Note: If both formats are used, they are merged. The new mcp_tools format takes precedence for any MCP listed in both.
Managing Agents
Via CLI

# List all agents
archiad agent list

# Show agent configuration
archiad agent show assistant

# Set/update an agent from a file
archiad agent set --name assistant --file ./my-agent.toml

# Remove an agent
archiad agent unset --name assistant

Via REST API

# List all agent configurations
curl http://localhost:8080/v1/agent/config

# Get specific agent
curl http://localhost:8080/v1/agent/config/assistant

# Create new agent
curl -X POST http://localhost:8080/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "helper",
    "model_name": "claude-haiku-4-5-20251001",
    "enabled": true,
    "description": "A helpful assistant",
    "system_prompt": "You are a helpful assistant."
  }'

# Update agent
curl -X PUT http://localhost:8080/v1/agent/config/helper \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": false
  }'

# Delete agent
curl -X DELETE http://localhost:8080/v1/agent/config/helper

Example Configurations
Customer Support Agent

# ~/.archia/agents/support.toml
name = "support"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
description = "Customer support agent with knowledge base access"

system_prompt = """
You are a friendly and professional customer support agent.
- Always greet customers warmly
- Be patient and understanding
- Escalate complex issues when appropriate
- Never share internal policies or workarounds
"""

[mcp_tools]
knowledge_base = ["search", "get_article"]
ticketing = ["create_ticket", "update_ticket"]

Code Review Agent

# ~/.archia/agents/code-reviewer.toml
name = "code-reviewer"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
description = "Automated code review assistant"
system_prompt_file = "code-review-prompt.md"

[mcp_tools]
github = ["get_pull_request", "list_files", "add_comment"]
filesystem = ["read_file"]

Research Agent with Sub-agent Management

# ~/.archia/agents/researcher.toml
name = "researcher"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
description = "Research coordinator that can delegate to specialized agents"
can_manage_agents = true

system_prompt = """
You are a research coordinator. You can:
1. Search the web for information
2. Delegate specialized tasks to other agents
3. Synthesize findings into comprehensive reports
"""

[mcp_tools]
web_search = null
document_store = ["save", "retrieve"]

Best Practices
1. Use Descriptive Names

# Good
name = "customer-support-tier1"
name = "code-review-python"
name = "data-analyst-financial"

# Avoid
name = "agent1"
name = "helper"
name = "bot"

2. Organize Prompt Files

~/.archia/
├── agents/
│   ├── support.toml
│   ├── analyst.toml
│   └── developer.toml
└── prompts/
    ├── support-prompt.md
    ├── analyst-prompt.md
    └── developer-prompt.md

3. Use Minimal Tool Access

Grant agents only the tools they need:

# Good - specific tools
[mcp_tools]
database = ["query"]  # Read-only

# Avoid - unnecessary access
[mcp_tools]
database = null  # All tools including write operations

4. Version Control Your Configurations

Store agent configurations in version control for:

    Change tracking
    Environment consistency
    Easy rollbacks
    Team collaboration

Migration from Legacy Config

If you’re migrating from the server config.toml format:

Old format (in config.toml):

[[agents]]
name = "assistant"
model = "claude-haiku-4-5-20251001"
system = "You are a helpful assistant"
mcps = ["database", "filesystem"]

New format (separate file):

# ~/.archia/agents/assistant.toml
name = "assistant"
model_name = "claude-haiku-4-5-20251001"
enabled = true
system_prompt = "You are a helpful assistant"

[mcp_tools]
database = null
filesystem = null
Tool Configuration Overview

Complete guide to defining and configuring MCP tools in Archia.
Configuration Options

Archia supports configuring tools in two ways:

    User Tools: Custom tools defined as TOML files in ~/.archia/tools/user/
    Marketplace Tools: Pre-packaged tools installed from the Archia marketplace

Tool Storage Location

Tools are stored in the tools directory:

    Production: ~/.archia/tools/
    Development: ~/.archia_dev/tools/

Directory Structure

~/.archia/tools/
├── user/                           # User-created tools
│   ├── my-custom-tool/
│   │   └── tool.toml
│   └── another-tool/
│       └── tool.toml
└── {organization}/                 # Marketplace tools
    └── {tool-name}/
        └── {version}/
            └── tool.toml

Tool Configuration Format

Each tool is defined in a tool.toml file with the following structure:
Basic Tool Definition

# ~/.archia/tools/user/my-tool/tool.toml

identifier = "my-tool"
name = "My Custom Tool"
description = "A custom MCP tool for specific tasks"
provider = "My Organization"
version = "1.0.0"

Local Tool (STDIO Transport)

Local tools run as processes on the same machine using STDIO communication:

identifier = "local-database"
name = "Local Database Tool"
description = "SQLite database access via MCP"
provider = "Archia"
version = "1.0.0"
type = "mcp"

[local]
cmd = "mcp-sqlite"
args = ["--database", "/data/mydb.sqlite"]
timeout_secs = 30

[local.env]
LOG_LEVEL = "info"
MAX_CONNECTIONS = "10"

Remote Tool (HTTP/SSE Transport)

Remote tools connect to external MCP servers over HTTP:

identifier = "remote-api"
name = "Remote API Tool"
description = "Connect to external API service"
provider = "External Provider"
version = "1.0.0"
type = "mcp"

[remote]
url = "https://api.example.com/mcp"
transport = "streaming_http"    # or "sse"
timeout_secs = 60

Configuration Fields Reference
Top-Level Fields
Field	Type	Required	Description
identifier	String	Yes	Unique identifier for the tool
version	String	No	Semantic version (default: “0.0.0”)
name	String	Yes	Human-readable display name
description	String	No	Description of the tool’s purpose
provider	String	No	Organization or author name
icon	String	No	URL or path to tool icon
type	String	No	Tool type (currently only “mcp”)
initial_install	Boolean	No	Auto-install on first run
Local Configuration ([local])
Field	Type	Required	Description
cmd	String	Yes	Executable command or path
args	Array	No	Command-line arguments
env	Map	No	Environment variables
timeout_secs	Integer	No	Tool call timeout (default: 30)
os	String	No	Target OS constraint
arch	String	No	Target architecture constraint
Remote Configuration ([remote])
Field	Type	Required	Description
url	String	Yes	MCP server URL
transport	String	Yes	“streaming_http” or “sse”
auth_type	String	No	“none”, “bearer”, or “oauth”
auth_token	String	No	Bearer token (when auth_type is “bearer”)
oauth_scopes	Array	No	OAuth scopes to request
oauth_client_id	String	No	OAuth client ID
oauth_client_secret	String	No	OAuth client secret
timeout_secs	Integer	No	Tool call timeout (default: 30)
os	String	No	Target OS constraint
arch	String	No	Target architecture constraint
Authentication Types
No Authentication

[remote]
url = "https://public-api.example.com/mcp"
transport = "streaming_http"
auth_type = "none"

Bearer Token Authentication

[remote]
url = "https://api.example.com/mcp"
transport = "streaming_http"
auth_type = "bearer"
auth_token = "your-api-key-here"

OAuth 2.1 Authentication

[remote]
url = "https://oauth-api.example.com/mcp"
transport = "streaming_http"
auth_type = "oauth"
oauth_scopes = ["mcp", "read", "write"]
oauth_client_id = "your-client-id"
oauth_client_secret = "your-client-secret"    # Optional for PKCE

Managing Tools
Via REST API

# List all tools
curl http://localhost:8080/v1/tool

# List only user tools
curl "http://localhost:8080/v1/tool?user_only=true"

# Get specific tool
curl http://localhost:8080/v1/tool/my-tool

# Create new tool
curl -X POST http://localhost:8080/v1/tool \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "new-tool",
    "name": "New Tool",
    "description": "A new MCP tool",
    "local": {
      "cmd": "my-mcp-server",
      "args": ["--config", "/path/to/config"],
      "timeout_secs": 30
    }
  }'

# Update tool
curl -X PUT http://localhost:8080/v1/tool/new-tool \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Updated description",
    "local": {
      "cmd": "my-mcp-server",
      "args": ["--config", "/new/path"],
      "timeout_secs": 60
    }
  }'

# Delete tool
curl -X DELETE http://localhost:8080/v1/tool/new-tool

Example Configurations
File System Access Tool

# ~/.archia/tools/user/filesystem/tool.toml

identifier = "filesystem"
name = "File System Access"
description = "Read and write files on the local system"
provider = "Archia"
version = "1.0.0"
type = "mcp"

[local]
cmd = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem@latest"]
timeout_secs = 30

[local.env]
ALLOWED_PATHS = "/home/user/documents,/home/user/projects"

GitHub Integration Tool

# ~/.archia/tools/user/github/tool.toml

identifier = "github"
name = "GitHub Integration"
description = "Interact with GitHub repositories"
provider = "Archia"
version = "1.0.0"
type = "mcp"

[local]
cmd = "npx"
args = ["-y", "@modelcontextprotocol/server-github@latest"]
timeout_secs = 60

[local.env]
GITHUB_TOKEN = "${GITHUB_TOKEN}"

Remote Database Service

# ~/.archia/tools/user/cloud-db/tool.toml

identifier = "cloud-db"
name = "Cloud Database"
description = "Access cloud-hosted database service"
provider = "Cloud Provider"
version = "1.0.0"
type = "mcp"

[remote]
url = "https://db.cloudprovider.com/mcp/v1"
transport = "streaming_http"
auth_type = "bearer"
auth_token = "${CLOUD_DB_API_KEY}"
timeout_secs = 120

OAuth-Protected API

# ~/.archia/tools/user/protected-api/tool.toml

identifier = "protected-api"
name = "Protected API Service"
description = "OAuth-protected external API"
provider = "Third Party"
version = "1.0.0"
type = "mcp"

[remote]
url = "https://api.thirdparty.com/mcp"
transport = "streaming_http"
auth_type = "oauth"
oauth_scopes = ["mcp", "data:read", "data:write"]
oauth_client_id = "your-oauth-client-id"
timeout_secs = 90

Connecting Tools to Agents

Once tools are configured, grant agents access via the mcp_tools field in agent configuration:

# ~/.archia/agents/developer.toml

name = "developer"
model_name = "claude-sonnet-4-5-20250929"
enabled = true
description = "Development assistant with tool access"

[mcp_tools]
filesystem = ["read_file", "write_file", "list_directory"]
github = ["list_repos", "get_file", "create_issue"]
cloud-db = null    # All tools

Best Practices
1. Use Environment Variables for Secrets

Never hardcode API keys or tokens:

[local.env]
API_KEY = "${MY_API_KEY}"

[remote]
auth_token = "${SERVICE_TOKEN}"

2. Set Appropriate Timeouts

Adjust timeouts based on expected operation duration:

[local]
timeout_secs = 30    # Fast operations

[remote]
timeout_secs = 120   # Network latency + processing

3. Constrain File Access

Limit filesystem tools to specific directories:

[local.env]
ALLOWED_PATHS = "/safe/directory/only"
READONLY = "true"

4. Version Your Tools

Use semantic versioning for tracking:

version = "1.2.3"

Troubleshooting
Tool Won’t Start

    Verify the command exists and is executable:

    which mcp-sqlite

    Check environment variables are set:

    echo $MY_API_KEY

    Test the tool manually:

    mcp-sqlite --database /path/to/db

Connection Refused (Remote Tools)

    Verify the URL is correct and accessible
    Check authentication credentials
    Ensure network/firewall allows the connection

Timeout Errors

Increase the timeout_secs value for slow operations or network latency.REST API Reference

Complete reference for the Archia server REST API endpoints.

    📖 Interactive Documentation: View the Swagger/OpenAPI Documentation for an interactive API explorer.

Overview

The Archia server exposes a REST API for managing agents, tools, chats, and system operations. All endpoints are prefixed with /v1/.

Base URL: http://localhost:8080/v1

Authentication: Most endpoints require authentication. Include your credentials in the request headers.
API Sections
Section	Description
Responses API	Recommended - Generate model responses with streaming and tool support
Supported Models	List of all supported models and their capabilities
Agents API	Manage agent configurations and chat sessions
Tools API	Configure and manage MCP tools
System API	System endpoints for health, metrics, and models
Quick Start
Generate a Response (Recommended)

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "input": "What is the capital of France?"
  }'

Use an Agent

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:assistant",
    "input": "Help me analyze this data"
  }'

List Available Agents

curl http://localhost:8080/v1/agent/config

List Available Tools

curl http://localhost:8080/v1/tool

Error Responses

All endpoints return standard error responses:

{
  "error": {
    "code": "not_found",
    "message": "Agent 'unknown' not found"
  }
}

Common HTTP Status Codes:
Code	Description
200	Success
201	Created
204	No Content (successful delete)
400	Bad Request (invalid input)
401	Unauthorized
404	Not Found
500	Internal Server Error
OpenAPI / Swagger Documentation

Interactive API documentation is generated from the source code using OpenAPI/Swagger.

View Interactive API Documentation (Redoc) →

To generate the documentation locally:

# Generate OpenAPI spec and HTML docs
cargo run --package manage -- document server

This generates:

    target/doc/archiad.json - OpenAPI specification (machine-readable)
    target/doc/archiad.html - Redoc HTML documentation (interactive)

You can also get just the OpenAPI JSON:

cargo run --package archiad -- document > archiad.json
Configuration

The Archia server (archiad) uses a TOML configuration file for server settings. Agent and tool configurations are managed separately as individual TOML files.
Server Configuration

The server configuration file is provided on the command line:

archiad serve /etc/archiad/config.toml

Network Settings

Configure the HTTP server binding:

[network]
host = "0.0.0.0"
port = 8371

Field	Description
host	The network interface to bind. Use 0.0.0.0 to bind all interfaces, or 127.0.0.1 for localhost only.
port	The port number. Common choices: 8371 (default), 80 (HTTP), 443 (HTTPS via reverse proxy).
Local Inference Settings

Configure local model inference (if using local LLMs):

[local_inference]
max_concurrent_sessions = 4

Field	Description
max_concurrent_sessions	Maximum number of concurrent local inference sessions.
Limits Settings

Configure default limits for chat requests:

[limits]
max_tool_calls = 10
timeout_ms = 300000

Field	Description
max_tool_calls	Default maximum tool calls per chat (0 disables the limit).
timeout_ms	Default chat timeout in milliseconds (0 disables the timeout).
Agent Configuration

Agents are configured as individual TOML files stored in the agents directory:

    Production: ~/.archia/agents/
    Development: ~/.archia_dev/agents/

Example Agent

Create a file at ~/.archia/agents/assistant.toml:

name = "assistant"
model_name = "claude-haiku-4-5-20251001"
enabled = true
description = "A helpful AI assistant"

system_prompt = """
You are a helpful, friendly assistant. Answer questions clearly and concisely.
"""

# Optional: Grant access to specific MCP tools
[mcp_tools]
filesystem = ["read_file", "list_directory"]

Managing Agents via CLI

# List all agents
archiad agent list

# Show agent configuration
archiad agent show assistant

# Add/update an agent from a file
archiad agent set --name assistant --file ./my-agent.toml

# Remove an agent
archiad agent unset --name assistant

Managing Agents via API

# List agents
curl http://localhost:8080/v1/agent/config

# Create agent
curl -X POST http://localhost:8080/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "helper",
    "model_name": "claude-haiku-4-5-20251001",
    "enabled": true,
    "system_prompt": "You are a helpful assistant."
  }'

# Update agent
curl -X PUT http://localhost:8080/v1/agent/config/helper \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Delete agent
curl -X DELETE http://localhost:8080/v1/agent/config/helper

For complete agent configuration options, see Agent Configuration.
Tool Configuration

Tools (MCP servers) are configured as TOML files in the tools directory:

    User tools: ~/.archia/tools/user/{tool-name}/tool.toml
    Marketplace tools: ~/.archia/tools/{org}/{tool}/{version}/tool.toml

Example Local Tool

Create a directory and tool file at ~/.archia/tools/user/database/tool.toml:

identifier = "database"
name = "Database Tool"
description = "SQLite database access"
version = "1.0.0"
type = "mcp"

[local]
cmd = "mcp-sqlite"
args = ["--database", "/data/production.db"]
timeout_secs = 30

[local.env]
LOG_LEVEL = "info"

Example Remote Tool

identifier = "cloud-api"
name = "Cloud API"
description = "Remote API access"
version = "1.0.0"
type = "mcp"

[remote]
url = "https://api.example.com/mcp"
transport = "streaming_http"
auth_type = "bearer"
auth_token = "${API_TOKEN}"
timeout_secs = 60

Managing Tools via API

# List all tools
curl http://localhost:8080/v1/tool

# List user tools only
curl "http://localhost:8080/v1/tool?user_only=true"

# Get specific tool
curl http://localhost:8080/v1/tool/database

# Create tool
curl -X POST http://localhost:8080/v1/tool \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "my-tool",
    "name": "My Tool",
    "local": {
      "cmd": "my-mcp-server",
      "args": ["--config", "/path/to/config"]
    }
  }'

# Delete tool
curl -X DELETE http://localhost:8080/v1/tool/my-tool

For complete tool configuration options, see Tool Configuration.
System Prompts

System prompts can be stored as separate files in the prompts directory:

    Production: ~/.archia/prompts/
    Development: ~/.archia_dev/prompts/

Reference prompt files in agent configuration:

# Instead of inline system_prompt
system_prompt_file = "analyst.md"

The server will load ~/.archia/prompts/analyst.md as the system prompt.
Environment Variables
Required Variables
Variable	Description
ARCHIA_LICENSE	Your Archia license key
ANTHROPIC_API_KEY	Anthropic API key (for Claude models)
Optional Variables
Variable	Description
OPENAI_API_KEY	OpenAI API key (for GPT models)
GOOGLE_API_KEY	Google API key (for Gemini models)
Example systemd Service

[Unit]
Description=Archia Daemon
After=network.target

[Service]
ExecStart=/usr/bin/archiad serve /etc/archiad/config.toml
Type=simple
Restart=on-failure
Environment="ARCHIA_LICENSE=your-license-key"
Environment="ANTHROPIC_API_KEY=your-anthropic-key"
Environment="OPENAI_API_KEY=your-openai-key"

[Install]
WantedBy=multi-user.target

Directory Structure Overview

~/.archia/
├── agents/                    # Agent configuration files
│   ├── assistant.toml
│   ├── researcher.toml
│   └── support.toml
├── tools/                     # Tool/MCP configurations
│   ├── user/                  # User-created tools
│   │   ├── database/
│   │   │   └── tool.toml
│   │   └── filesystem/
│   │       └── tool.toml
│   └── archia/                # Marketplace tools
│       └── web-search/
│           └── 1.0.0/
│               └── tool.toml
├── prompts/                   # System prompt files
│   ├── assistant.md
│   └── researcher.md
├── models/                    # Local model files (if using local inference)
├── supported_models.json      # Cached model registry
└── supported_tools.json       # Cached tool registry

Configuration Validation

Common configuration errors:
Error	Cause	Solution
“Agent name invalid”	Name contains special characters	Use only letters, numbers, hyphens, underscores
“Model not found”	Invalid model_name	Check supported models via API
“MCP not found”	Tool identifier doesn’t exist	Create tool configuration first
“License invalid”	Missing/incorrect license	Set ARCHIA_LICENSE environment variableResponses API

The Responses API is the recommended endpoint for generating model responses. It provides a modern, flexible interface compatible with OpenAI’s API format, supporting streaming, tool calling, and agent routing.

    Recommended: Use the Responses API for all new integrations. It offers the most complete feature set and best developer experience.

Create Response

Generates a model response for the given input.

POST /v1/responses

Request Body

{
  "model": "agent:assistant",
  "input": "What can you help me with today?"
}

Limits and Timeouts

You can override server limits via metadata:

{
  "model": "claude-opus-4-5-20251101",
  "input": "Summarize the last three games",
  "metadata": {
    "tool_limits": { "max_tool_calls": 8 },
    "timeout_ms": 120000
  }
}

Resolution order:

    metadata.tool_limits.max_tool_calls → request max_tool_calls → server config defaults
    metadata.timeout_ms → server config defaults

Parameters
Field	Type	Required	Description
model	string	Yes	Model ID (see Supported Models), agent:{agent_name} for agent routing, or agent:{agent_name}:{model_override} to override an agent’s model
input	string/array	No	Text input or array of input items
instructions	string	No	System prompt / developer message
stream	boolean	No	Enable streaming responses (default: false)
max_output_tokens	integer	No	Maximum tokens to generate
store	boolean	No	Store the response for later retrieval
metadata	object	No	Arbitrary JSON metadata echoed back in the response
previous_response_id	string	No	Chain responses in a conversation
reasoning	object	No	Enable extended thinking/reasoning (see Reasoning)
tools	array	No	Tools the model may call (function or MCP tools)
tool_choice	string/object	No	How model selects tools
parallel_tool_calls	boolean	No	Allow parallel tool calls
max_tool_calls	integer	No	Maximum number of tool calls (fallback if not provided in metadata)
Reasoning

The reasoning parameter enables extended thinking capabilities for supported models. When enabled, the model will perform additional reasoning steps before generating its response, which can improve quality for complex tasks.
Basic Usage

{
  "model": "claude-sonnet-4-5-20250929",
  "input": "Solve this step by step: If a train travels 120 miles in 2 hours, then stops for 30 minutes, then travels another 90 miles in 1.5 hours, what is the average speed for the entire journey?",
  "reasoning": {
    "effort": "medium"
  }
}

Reasoning Parameters
Field	Type	Required	Description
effort	string	Yes	Reasoning intensity: "none", "low", "medium", or "high"
Effort Levels
Level	Description
none	Disable reasoning (supported by OpenAI gpt-5.0+)
low	Light reasoning, suitable for simpler problems
medium	Balanced reasoning for most tasks
high	Maximum reasoning depth for complex problems

    Note: If the reasoning parameter is omitted, the provider’s default behavior is used. For OpenAI gpt-5.1+, the default is "none" (no reasoning).

Supported Models

Reasoning is supported on models with the reasoning capability:

    Anthropic: Claude Sonnet 3.7+, Claude Sonnet 4+, Claude Opus 4+
    Google: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 3.0 Pro
    OpenAI: GPT-5.x, o-series models (o1, o3, o4)

    Note: GPT-4.x models do not support the reasoning parameter and will return an error if it’s provided. The reasoning parameter is silently ignored for unsupported models.

Provider-Specific Behavior

Different providers implement reasoning differently:
Provider	none	low	medium	high	Default (not specified)
OpenAI (gpt-5.0+)	No reasoning	Minimal reasoning	Balanced reasoning	Maximum reasoning	none (gpt-5.1)
gpt-oss (local)	Maps to low	Low thinking	Medium thinking	High thinking	medium
Anthropic	Disables thinking	~1K token budget	~8K token budget	~24K token budget	No thinking
Google	Maps to low	Low budget	Medium budget	High budget	Provider default

    Note: gpt-oss models don’t support fully disabling reasoning - "none" maps to "low" (minimal reasoning).

Example with Streaming

{
  "model": "claude-sonnet-4-5-20250929",
  "input": "Explain the proof of the Pythagorean theorem",
  "reasoning": {
    "effort": "high"
  },
  "stream": true
}

When streaming with reasoning enabled, you’ll receive response.reasoning_summary_text.delta events containing the model’s reasoning process, followed by the regular response content.
Direct Model Calls

You can call models directly by specifying the model ID and optionally including MCP tools inline:

{
  "model": "gpt-5.2",
  "input": [
    {
      "role": "user",
      "content": "Roll 2d4+1"
    }
  ],
  "tools": [
    {
      "type": "mcp",
      "server_label": "dmcp",
      "server_description": "A Dungeons and Dragons MCP server to assist with dice rolling.",
      "server_url": "https://dmcp-server.deno.dev/sse",
      "require_approval": "never"
    }
  ]
}

MCP Tool Parameters
Field	Type	Required	Description
type	string	Yes	Must be "mcp"
server_label	string	Yes	Identifier for the MCP server
server_description	string	No	Description of what the server provides
server_url	string	Yes	URL of the MCP server (SSE endpoint)
require_approval	string	No	Approval mode: "never", "always", or "auto"

This approach is useful when you want to:

    Use a specific model without agent configuration
    Dynamically specify MCP tools per request
    Test new tools without modifying agent config

Agent Routing

The recommended way to use the Responses API is through agent routing. Use the model field to route requests to configured agents:

{
  "model": "agent:assistant",
  "input": "Help me with my task"
}

This routes to the agent named “assistant” and uses its configured model, system prompt, and tool access.

Benefits of agent routing:

    Pre-configured system prompts
    Automatic MCP tool access
    Centralized agent management
    No need to specify model or instructions per request

Model Override

You can override an agent’s configured model while still using its system prompt and tools by appending the model name:

agent:{agent_name}:{model_override}

Examples:

// Use agent's default model
{
  "model": "agent:assistant",
  "input": "Hello!"
}

// Override with Claude
{
  "model": "agent:assistant:claude-haiku-4-5-20251001",
  "input": "Hello!"
}

// Override with gpt-5.2
{
  "model": "agent:assistant:gpt-5.2",
  "input": "Hello!"
}

This is useful when you want to:

    Test an agent’s prompts and tools with different models
    Use a faster/cheaper model for simple tasks
    Use a more capable model for complex tasks
    A/B test model performance with the same agent configuration

Response Format
Non-Streaming Response

{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1705312200,
  "status": "completed",
  "model": "claude-sonnet-4-5-20250929",
  "output": [
    {
      "type": "message",
      "id": "msg_xyz789",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The capital of France is Paris."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 25,
    "output_tokens": 12,
    "total_tokens": 37
  }
}

Response with Reasoning

When reasoning is enabled, the response includes a reasoning output item before the message:

{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1705312200,
  "status": "completed",
  "model": "claude-sonnet-4-5-20250929",
  "output": [
    {
      "type": "reasoning",
      "id": "reasoning_def456",
      "status": "completed",
      "summary": [
        {
          "type": "summary_text",
          "text": "To solve this problem, I need to calculate the total distance and total time..."
        }
      ]
    },
    {
      "type": "message",
      "id": "msg_xyz789",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The average speed for the entire journey is 42 mph."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 45,
    "output_tokens": 156,
    "total_tokens": 201
  }
}

Response Fields
Field	Type	Description
id	string	Unique response identifier
object	string	Always “response”
created_at	integer	Unix timestamp of creation
status	string	One of: completed, failed, in_progress, cancelled
model	string	Model used for generation
output	array	Array of output items (messages, reasoning, function calls)
usage	object	Token usage statistics
error	object	Error details if status is “failed”
Output Item Types
Type	Description
message	Assistant’s response message with text content
reasoning	Model’s reasoning/thinking process (when reasoning enabled)
function_call	A tool/function call made by the model
function_call_output	Result from a tool/function call
Streaming

When stream: true, the endpoint returns Server-Sent Events (SSE):

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "agent:assistant",
    "input": "Tell me a story",
    "stream": true
  }'

Event Types

event: response.created
data: {"id":"resp_abc123","object":"response","status":"in_progress",...}

event: response.output_item.added
data: {"type":"message","id":"msg_xyz789","role":"assistant",...}

event: response.content_part.added
data: {"type":"output_text","text":""}

event: response.output_text.delta
data: {"delta":"Once upon"}

event: response.output_text.delta
data: {"delta":" a time..."}

event: response.output_text.done
data: {"text":"Once upon a time..."}

Reasoning Events (when reasoning is enabled)

When reasoning is enabled, additional events are sent before the main response content:

event: response.reasoning_summary_part.added
data: {"item_id":"reasoning_abc","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"item_id":"reasoning_abc","output_index":0,"summary_index":0,"delta":"Let me think through this..."}

event: response.reasoning_summary_text.delta
data: {"item_id":"reasoning_abc","output_index":0,"summary_index":0,"delta":" First, I need to consider..."}

event: response.reasoning_summary_text.done
data: {"item_id":"reasoning_abc","output_index":0,"summary_index":0,"text":"Let me think through this... First, I need to consider..."}

event: response.output_item.done data: {“type”:“message”,“id”:“msg_xyz789”,“status”:“completed”,…}

event: response.completed data: {“id”:“resp_abc123”,“status”:“completed”,“usage”:{…}}


### Event Sequence

Standard sequence:

1. `response.created` - Response object created
2. `response.output_item.added` - New output item (message or function call)
3. `response.content_part.added` - New content part added
4. `response.output_text.delta` - Text chunk (repeated)
5. `response.output_text.done` - Text content complete
6. `response.output_item.done` - Output item complete
7. `response.completed` - Full response complete

With reasoning enabled, reasoning events appear after `response.created` and before the main content:

1. `response.created`
2. `response.reasoning_summary_part.added` - Reasoning output started
3. `response.reasoning_summary_text.delta` - Reasoning text chunk (repeated)
4. `response.reasoning_summary_text.done` - Reasoning complete
5. `response.output_item.added` - Main response content begins
6. ... (standard content events)
7. `response.completed`

---

## Conversation Chaining

Chain multiple responses together using `previous_response_id` to maintain conversation context:

```bash
# First message
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:assistant",
    "input": "What is machine learning?"
  }'
# Response includes "id": "resp_abc123"

# Follow-up message
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:assistant",
    "previous_response_id": "resp_abc123",
    "input": "Can you give me a specific example?"
  }'

Examples
Chat with an Agent

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:assistant",
    "input": "Hello! What can you help me with?"
  }'

Agent with Model Override

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:assistant:gpt-5.2",
    "input": "Hello! What can you help me with?"
  }'

Streaming Chat

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "agent:assistant",
    "input": "Explain how APIs work",
    "stream": true
  }'

Multi-turn Conversation

# Ask a question
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:researcher",
    "input": "What are the main causes of climate change?"
  }'

# Follow up (using the response ID from above)
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:researcher",
    "previous_response_id": "resp_abc123",
    "input": "What solutions are being proposed?"
  }'

Direct Model with MCP Tools

curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model": "gpt-5.2",
    "input": [
      {
        "role": "user",
        "content": "Roll 2d4+1 for damage"
      }
    ],
    "tools": [
      {
        "type": "mcp",
        "server_label": "dmcp",
        "server_description": "A D&D MCP server for dice rolling",
        "server_url": "https://dmcp-server.deno.dev/sse",
        "require_approval": "never"
      }
    ]
  }'

Using the OpenAI SDK

The Responses API is compatible with the OpenAI SDK:

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",  # Archia uses Basic auth
    default_headers={"Authorization": "Basic <credentials>"}
)

response = client.responses.create(
    model="agent:assistant",
    input="What's the weather like today?"
)

print(response.output[0].content[0].text)

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-used",
  defaultHeaders: { Authorization: "Basic <credentials>" },
});

const response = await client.responses.create({
  model: "agent:assistant",
  input: "What's the weather like today?",
});

console.log(response.output[0].content[0].text);

Langfuse Integration

Langfuse provides observability for LLM applications. You can trace Archia API calls to monitor performance, debug issues, and analyze usage.
Python with Langfuse

from openai import OpenAI
from langfuse import Langfuse

# Initialize clients
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",
    default_headers={"Authorization": "Basic <credentials>"}
)

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000"
)

# Create a trace
trace = langfuse.trace(
    name="chat-with-agent",
    input={"prompt": "Hello!"},
    tags=["archia", "assistant"]
)

# Create a generation span
generation = trace.generation(
    name="responses-api-call",
    model="agent:assistant",
    input="Hello!"
)

# Make the API call
response = client.responses.create(
    model="agent:assistant",
    input="Hello!"
)

# Extract output and complete the trace
output_text = response.output[0].content[0].text
generation.end(
    output=output_text,
    usage={
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
        "total": response.usage.total_tokens
    }
)

# Flush traces
langfuse.flush()

TypeScript with Langfuse

import OpenAI from "openai";
import Langfuse from "langfuse";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-used",
  defaultHeaders: { Authorization: "Basic <credentials>" },
});

const langfuse = new Langfuse({
  publicKey: "pk-lf-...",
  secretKey: "sk-lf-...",
  baseUrl: "http://localhost:3000",
});

// Create a trace
const trace = langfuse.trace({
  name: "chat-with-agent",
  input: { prompt: "Hello!" },
  tags: ["archia", "assistant"],
});

// Create a generation span
const generation = trace.generation({
  name: "responses-api-call",
  model: "agent:assistant",
  input: "Hello!",
});

// Make the API call
const response = await client.responses.create({
  model: "agent:assistant",
  input: "Hello!",
});

// Extract output and complete the trace
const outputText = response.output[0].content[0].text;
generation.end({
  output: outputText,
  usage: {
    input: response.usage.input_tokens,
    output: response.usage.output_tokens,
    total: response.usage.total_tokens,
  },
});

// Flush traces
await langfuse.flushAsync();

Python with Langfuse Annotations

Using the @observe decorator for automatic tracing:

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe

# Initialize clients
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",
    default_headers={"Authorization": "Basic <credentials>"}
)

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000"
)

@observe(name="chat_with_agent")
def chat_with_agent(prompt: str, agent: str = "assistant") -> str:
    """Chat with an agent and return the response."""
    response = client.responses.create(
        model=f"agent:{agent}",
        input=prompt
    )
    
    output_text = response.output[0].content[0].text
    return output_text

@observe(name="multi_turn_conversation")
def multi_turn_conversation(messages: list[dict]) -> str:
    """Have a multi-turn conversation with an agent."""
    previous_response_id = None
    
    for msg in messages:
        if previous_response_id:
            response = client.responses.create(
                model="agent:assistant",
                previous_response_id=previous_response_id,
                input=msg["content"]
            )
        else:
            response = client.responses.create(
                model="agent:assistant",
                input=msg["content"]
            )
        
        previous_response_id = response.id
    
    return response.output[0].content[0].text

@observe(name="direct_model_call_with_tools")
def direct_model_call_with_tools(prompt: str) -> str:
    """Call a model directly with MCP tools."""
    response = client.responses.create(
        model="gpt-5.2",
        input=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "mcp",
                "server_label": "dmcp",
                "server_description": "A Dungeons and Dragons MCP server",
                "server_url": "https://dmcp-server.deno.dev/sse",
                "require_approval": "never"
            }
        ]
    )
    
    return response.output[0].content[0].text

# Usage examples
if __name__ == "__main__":
    # Simple chat
    result = chat_with_agent("What is machine learning?")
    print(result)
    
    # Multi-turn conversation
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "user", "content": "Can you give me a specific example?"}
    ]
    result = multi_turn_conversation(messages)
    print(result)
    
    # Direct model call with tools
    result = direct_model_call_with_tools("Roll 2d4+1 for damage")
    print(result)
    
    # Flush traces to Langfuse
    langfuse.flush()

The @observe decorator automatically:

    Creates a trace for each function call
    Captures input and output
    Measures execution time
    Logs any errors that occur
    Tracks nested function calls as child spans

What Langfuse Captures
Field	Description
Model	Agent name (e.g., agent:assistant)
Input	The prompt sent to the API
Output	The response text
Usage	Token counts (input, output, total)
Tags	Filterable tags for organizing traces
Latency	Request duration
Metadata	Custom context and attributes

For complete examples, see the poc/shottracker/langfuse/ directory which includes full Python and TypeScript implementations.
Error Handling
Error Response

{
  "id": "resp_abc123",
  "status": "failed",
  "error": {
    "error_type": "invalid_request",
    "message": "Agent 'unknown-agent' not found"
  }
}

Common Errors
Error Type	Description
invalid_request	Malformed request or invalid parameters
agent_not_found	Agent routing failed - agent doesn’t exist
rate_limit_exceeded	Too many requests
context_length_exceeded	Input too long for model
Agents API

Manage agent configurations and chat sessions.
Agent Configuration
List Agent Configurations

Returns all configured agents.

GET /v1/agent/config

Response:

{
  "configs": [
    {
      "name": "assistant",
      "description": "A helpful AI assistant",
      "model_name": "claude-haiku-4-5-20251001",
      "enabled": true,
      "mcp_names": [],
      "mcp_tools": {
        "filesystem": [
          "read_file"
        ]
      },
      "system_prompt": "You are a helpful assistant.",
      "system_prompt_file": null,
      "can_manage_agents": false,
      "enable_extended_thinking": false,
      "reasoning_effort": null
    }
  ]
}

Get Agent Configuration

Returns configuration for a specific agent.

GET /v1/agent/config/{name}

Path Parameters:
Parameter	Type	Description
name	string	The agent name

Response: Single agent configuration object.
Create Agent Configuration

Creates a new agent configuration.

POST /v1/agent/config

Request Body:

{
  "name": "new-agent",
  "model_name": "claude-sonnet-4-5-20250929",
  "enabled": true,
  "description": "A new agent",
  "mcp_names": [],
  "mcp_tools": {
    "database": null
  },
  "system_prompt": "You are an expert assistant.",
  "system_prompt_file": null,
  "can_manage_agents": false,
  "reasoning_effort": "medium"
}

Parameters:
Field	Type	Required	Description
name	string	Yes	Unique agent identifier
model_name	string	Yes	LLM model to use
enabled	boolean	Yes	Whether agent is active
description	string	No	Human-readable description
mcp_names	array	No	Legacy MCP access list (deprecated)
mcp_tools	object	No	MCP name → tool list mapping
system_prompt	string	No	Inline system prompt
system_prompt_file	string	No	Reference to prompt file in ~/.archia/prompts/
can_manage_agents	boolean	No	Allow agent to spawn/manage other agents
enable_extended_thinking	boolean	No	Deprecated: Use reasoning_effort instead. If true, sets reasoning to "medium"
reasoning_effort	string	No	Reasoning intensity: "none", "low", "medium", or "high". Takes precedence over enable_extended_thinking
Reasoning Effort

The reasoning_effort field controls the model’s reasoning behavior. See the Responses API documentation for details on how different providers handle each level.
Level	Description
none	Disable reasoning (supported by OpenAI gpt-5.0+; maps to low for gpt-oss)
low	Light reasoning, suitable for simpler problems
medium	Balanced reasoning for most tasks
high	Maximum reasoning depth for complex problems

    Note: If neither reasoning_effort nor enable_extended_thinking is set, the provider’s default behavior is used.

Response: 201 Created with the created agent configuration.
Update Agent Configuration

Updates an existing agent configuration.

PUT /v1/agent/config/{name}

Path Parameters:
Parameter	Type	Description
name	string	The agent name to update

Request Body: Same fields as create, all optional.

Response: 200 OK with the updated agent configuration.
Delete Agent Configuration

Deletes an agent configuration.

DELETE /v1/agent/config/{name}

Path Parameters:
Parameter	Type	Description
name	string	The agent name to delete

Response: 204 No Content
List Available Agents

Returns a list of enabled agents that can be used for chat.

GET /v1/agent

Query Parameters:
Parameter	Type	Description
name	string	Filter by agent name
mcp	string	Filter by MCP access

Response:

{
  "agents": [
    {
      "name": "assistant",
      "mcps": [
        "filesystem",
        "database"
      ]
    }
  ]
}

Agent Chat
Create Agent Chat

Creates a new chat session with an agent.

POST /v1/agent/chat

Request Body:

{
  "agent_name": "assistant",
  "parent_chat_id": 1,
  "initial_message": "Hello, I need help with..."
}

Parameters:
Field	Type	Required	Description
agent_name	string	Yes	Agent to chat with
parent_chat_id	integer	Yes	Parent chat context
initial_message	string	No	Optional first message

Response:

{
  "agent_chat_id": 42,
  "agent_name": "assistant",
  "parent_chat_id": 1,
  "model_name": "claude-haiku-4-5-20251001",
  "agent_management_enabled": false
}

Chat Messages
Get Messages

Retrieves messages from a chat.

GET /v1/chat/{chat_id}/message

Path Parameters:
Parameter	Type	Description
chat_id	integer	The chat ID

Response: Array of message objects.
Send Message

Sends a message to a chat.

POST /v1/chat/{chat_id}/message

Path Parameters:
Parameter	Type	Description
chat_id	integer	The chat ID

Request Body:

{
  "agent": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello, how can you help me?"
    }
  ]
}

Response:

{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'm here to help! What would you like to know?"
    }
  ],
  "error": null
}

Delete Chat

Deletes a chat session.

DELETE /v1/chat/{chat_id}

Path Parameters:
Parameter	Type	Description
chat_id	integer	The chat ID to delete

Response: 204 No Content
Agent Management

Control agent management capabilities for chat sessions.
Enable Agent Management

Enables agent management capabilities for a chat, allowing the agent to spawn and manage other agents.

POST /v1/agent/management/{chat_id}/enable

Path Parameters:
Parameter	Type	Description
chat_id	integer	The chat ID

Response: 200 OK
Disable Agent Management

Disables agent management capabilities for a chat.

POST /v1/agent/management/{chat_id}/disable

Path Parameters:
Parameter	Type	Description
chat_id	integer	The chat ID

Response: 200 OK
Agent Monitoring
List Active Agents

Returns currently active agent sessions.

GET /v1/agent/monitoring/active

Response:

{
  "active_agents": [
    {
      "chat_id": 1,
      "user_data": {},
      "pending": false,
      "has_messages": true,
      "last_model": "claude-haiku-4-5-20251001",
      "agent_management_enabled": false,
      "message_count": 5
    }
  ],
  "total": 1
}

List Waiting Agents

Returns agent sessions waiting for input (e.g., waiting for tool results).

GET /v1/agent/monitoring/waiting

Response:

{
  "waiting_agents": [
    {
      "agent_chat_id": 42,
      "parent_chat_id": 1,
      "inflight_id": "abc123"
    }
  ],
  "total": 1
}

Examples
Create and Use an Agent

# Create agent
curl -X POST http://localhost:8080/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "helper",
    "model_name": "claude-haiku-4-5-20251001",
    "enabled": true,
    "system_prompt": "You are a helpful assistant."
  }'

# Use via Responses API (recommended)
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:helper",
    "input": "What can you help me with?"
  }'

Create Agent with Reasoning

# Create agent with high reasoning for complex tasks
curl -X POST http://localhost:8080/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "reasoning-agent",
    "model_name": "gpt-5.1",
    "enabled": true,
    "system_prompt": "You are a problem-solving assistant.",
    "reasoning_effort": "high"
  }'

# Create agent with no reasoning for fast responses
curl -X POST http://localhost:8080/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fast-agent",
    "model_name": "gpt-5.1",
    "enabled": true,
    "system_prompt": "You are a quick assistant.",
    "reasoning_effort": "none"
  }'

List and Filter Agents

# List all agent configs
curl http://localhost:8080/v1/agent/config

# List enabled agents
curl http://localhost:8080/v1/agent

# Filter by MCP access
curl "http://localhost:8080/v1/agent?mcp=database"

Next Steps

    Responses API → - Use agents via the Responses API (recommended)
    Tools API → - Configure MCP tools for agents
    Agent Configuration → - File-based agent configuration

Tools API

Manage MCP tool configurations.
List Tools

Returns all available MCP tools.

GET /v1/tool

Query Parameters:
Parameter	Type	Description
user_only	boolean	Return only user-created tools (default: false)

Response:

{
  "tools": [
    {
      "identifier": "database",
      "version": "1.0.0",
      "name": "Database Tool",
      "description": "SQLite database access",
      "provider": "Archia",
      "icon": null,
      "tool_type": "mcp",
      "local": {
        "cmd": "mcp-sqlite",
        "args": ["--database", "/data/db.sqlite"],
        "env": {},
        "timeout_secs": 30
      },
      "remote": null
    }
  ]
}

Get Tool

Returns a specific tool by identifier.

GET /v1/tool/{identifier}

Path Parameters:
Parameter	Type	Description
identifier	string	Tool identifier (e.g., org/tool or my-tool)

Response: Single tool object.

Example:

curl http://localhost:8080/v1/tool/database

Create Tool

Creates a new tool configuration.

POST /v1/tool

Local Tool

For tools running locally via STDIO:

Request Body:

{
  "identifier": "my-tool",
  "name": "My Tool",
  "description": "A custom MCP tool",
  "version": "1.0.0",
  "provider": "My Organization",
  "local": {
    "cmd": "my-mcp-server",
    "args": ["--config", "/path/to/config"],
    "env": {"API_KEY": "secret"},
    "timeout_secs": 30
  }
}

Remote Tool

For tools connecting to external MCP servers:

Request Body:

{
  "identifier": "remote-tool",
  "name": "Remote Tool",
  "description": "External MCP service",
  "remote": {
    "url": "https://api.example.com/mcp",
    "transport": "streaming_http",
    "auth_type": "bearer",
    "auth_token": "your-token",
    "timeout_secs": 60
  }
}

Parameters
Field	Type	Required	Description
identifier	string	Yes	Unique tool identifier
name	string	Yes	Human-readable display name
description	string	No	Description of the tool’s purpose
version	string	No	Semantic version (default: “0.0.0”)
provider	string	No	Organization or author name
icon	string	No	URL or path to tool icon
local	object	No*	Local tool configuration
remote	object	No*	Remote tool configuration

*Either local or remote must be provided.
Local Configuration
Field	Type	Required	Description
cmd	string	Yes	Executable command or path
args	array	No	Command-line arguments
env	object	No	Environment variables
timeout_secs	integer	No	Tool call timeout (default: 30)
Remote Configuration
Field	Type	Required	Description
url	string	Yes	MCP server URL
transport	string	Yes	“streaming_http” or “sse”
auth_type	string	No	“none”, “bearer”, or “oauth”
auth_token	string	No	Bearer token (when auth_type is “bearer”)
oauth_scopes	array	No	OAuth scopes to request
oauth_client_id	string	No	OAuth client ID
oauth_client_secret	string	No	OAuth client secret
timeout_secs	integer	No	Tool call timeout (default: 30)

Response: 201 Created with the created tool.
Update Tool

Updates an existing tool configuration.

PUT /v1/tool/{identifier}

Path Parameters:
Parameter	Type	Description
identifier	string	Tool identifier to update

Request Body: Same fields as create, all optional.

Response: 200 OK with the updated tool.

Example:

curl -X PUT http://localhost:8080/v1/tool/my-tool \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Updated description",
    "local": {
      "cmd": "my-mcp-server",
      "args": ["--config", "/new/path"],
      "timeout_secs": 60
    }
  }'

Delete Tool

Deletes a tool configuration.

DELETE /v1/tool/{identifier}

Path Parameters:
Parameter	Type	Description
identifier	string	Tool identifier to delete

Response: 204 No Content

Example:

curl -X DELETE http://localhost:8080/v1/tool/my-tool

Examples
Create a Database Tool

curl -X POST http://localhost:8080/v1/tool \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "sqlite-db",
    "name": "SQLite Database",
    "description": "Query SQLite databases",
    "local": {
      "cmd": "mcp-sqlite",
      "args": ["--database", "/data/analytics.db"],
      "timeout_secs": 30
    }
  }'

Create a GitHub Tool

curl -X POST http://localhost:8080/v1/tool \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "github",
    "name": "GitHub Integration",
    "description": "Interact with GitHub repositories",
    "local": {
      "cmd": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github@latest"],
      "env": {"GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"},
      "timeout_secs": 60
    }
  }'

Create a Remote API Tool

curl -X POST http://localhost:8080/v1/tool \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "cloud-service",
    "name": "Cloud Service API",
    "description": "Access cloud-hosted service",
    "remote": {
      "url": "https://api.cloudservice.com/mcp",
      "transport": "streaming_http",
      "auth_type": "bearer",
      "auth_token": "your-api-key",
      "timeout_secs": 120
    }
  }'

List Only User Tools

curl "http://localhost:8080/v1/tool?user_only=true"

Granting Tool Access to Agents

After creating tools, grant agents access via the agent configuration:

curl -X PUT http://localhost:8080/v1/agent/config/assistant \
  -H "Content-Type: application/json" \
  -d '{
    "mcp_tools": {
      "sqlite-db": ["query", "list_tables"],
      "github": null
    }
  }'

    null or [] grants access to all tools from that MCP
    An array of strings grants access to only those specific tools

System API

System endpoints for health checks, metrics, and model information.
Models
List Models

Returns all available LLM models.

GET /v1/models

Response:

{
  "models": [
    {
      "name": "claude-sonnet-4-5-20250929",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 8192,
      "supports_tools": true,
      "supports_vision": true
    },
    {
      "name": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 8192,
      "supports_tools": true,
      "supports_vision": true
    }
  ]
}

Get Model

Returns details for a specific model.

GET /v1/models/{name}

Path Parameters:
Parameter	Type	Description
name	string	Model name (e.g., “claude-sonnet-4-5-20250929”)

Response:

{
  "name": "claude-sonnet-4-5-20250929",
  "provider": "anthropic",
  "context_window": 200000,
  "max_output_tokens": 8192,
  "supports_tools": true,
  "supports_vision": true
}

Error Response (404):

{
  "error": {
    "code": "not_found",
    "message": "Model 'unknown-model' not found"
  }
}

System Time
Get Server Time

Returns the current server time. Useful for synchronization and debugging.

GET /v1/system/time

Response:

{
  "time": "2024-01-15T10:30:00Z"
}

Metrics
Get Metrics

Returns server metrics in Prometheus format.

GET /v1/metrics

Response:

# HELP archia_requests_total Total number of requests
# TYPE archia_requests_total counter
archia_requests_total{endpoint="/v1/responses"} 1523
archia_requests_total{endpoint="/v1/agent/config"} 42

# HELP archia_active_chats Number of active chat sessions
# TYPE archia_active_chats gauge
archia_active_chats 7

# HELP archia_request_duration_seconds Request duration in seconds
# TYPE archia_request_duration_seconds histogram
archia_request_duration_seconds_bucket{le="0.1"} 1200
archia_request_duration_seconds_bucket{le="0.5"} 1450
archia_request_duration_seconds_bucket{le="1.0"} 1510
archia_request_duration_seconds_bucket{le="+Inf"} 1523

Use with Prometheus:

Add to your prometheus.yml:

scrape_configs:
  - job_name: 'archia'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/v1/metrics'

Examples
Check Server Health

# Quick health check using time endpoint
curl http://localhost:8080/v1/system/time

List Available Models

curl http://localhost:8080/v1/models

Get Specific Model Info

curl http://localhost:8080/v1/models/claude-sonnet-4-5-20250929

Scrape Metrics

curl http://localhost:8080/v1/metrics

Library Bindings

Embed Archia directly into your applications with native language bindings.
Overview

Archia’s library bindings allow you to integrate the full power of MCP agents directly into your applications without running a separate server or introducing a new dependency. This provides:

    Zero Network Overhead: Direct in-process communication
    Native Integration: Idiomatic APIs for each language
    Embedded Deployment: Perfect for desktop apps, CLI tools, and edge devices
    Full Control: Direct access to the Archia engine

C/C++ Integration

Native bindings for system programming and embedded applications.
Overview

The C/C++ bindings provide direct access to the Archia engine through a Foreign Function Interface (FFI). This is the foundation for all other language bindings and offers maximum performance with minimal overhead.

Key Features:

    Zero-overhead FFI interface
    Thread-safe operations
    Manual memory management for precise control
    Compatible with C99 and C++11 or later
    Header-only integration

Quick Start
Basic Example

#include <archia.h>
#include <stdio.h>

int main() {
    // Initialize engine
    Engine* engine = archia_new();
    if (!engine) {
        fprintf(stderr, "Failed to create engine\n");
        return 1;
    }

    // Create chat session
    size_t chat_id = archia_new_chat(
        engine,
        "You are a helpful assistant",
        NULL, 0,  // No MCP servers
        NULL      // No user data
    );

    // Send message
    Error* err = archia_send(
        engine,
        chat_id,
        "claude-haiku-4-5-20251001",
        "What is the capital of France?"
    );

    if (err) {
        fprintf(stderr, "Error: %s\n", archia_error_message(err));
        archia_error_free(err);
        archia_free(engine);
        return 1;
    }

    // Wait for response
    archia_wait_on(engine, chat_id);

    // Get response
    size_t msg_count = archia_chat_len(engine, chat_id);
    const char* response = archia_chat_message(engine, chat_id, msg_count - 1);
    
    printf("Assistant: %s\n", response);

    // Cleanup
    archia_free(engine);
    return 0;
}

Compilation

# Linux
gcc -o myapp myapp.c -larchia -lpthread

# macOS
clang -o myapp myapp.c -larchia

# Windows (MSVC)
cl myapp.c archia.lib

# Windows (MinGW)
gcc -o myapp.exe myapp.c -larchia -lws2_32

Advanced Usage
With MCP Servers

#include <archia.h>

int main() {
    Engine* engine = archia_new();
    
    // Configure MCP server
    MCPConfig database = {
        .name = "database",
        .cmd = "mcp-sqlite",
        .args = {"production.db", NULL},
        .env = {
            "LOG_LEVEL=info",
            "MAX_CONNECTIONS=10",
            NULL
        }
    };
    
    // Add MCP to engine
    size_t mcp_id = archia_add_mcp(engine, &database);
    
    // Create chat with MCP access
    size_t mcps[] = {mcp_id};
    size_t chat_id = archia_new_chat(
        engine,
        "You are a data analyst with database access",
        mcps, 1,  // 1 MCP server
        NULL
    );
    
    // Query using MCP tools
    archia_send(engine, chat_id, "claude-3-5-sonnet",
        "Show me top 10 customers by revenue");
    
    archia_wait_on(engine, chat_id);
    // ... handle response ...
    
    archia_free(engine);
    return 0;
}

Async Operations

#include <archia.h>
#include <pthread.h>

typedef struct {
    Engine* engine;
    size_t chat_id;
} ThreadData;

void* worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // Each thread can send to different chats
    archia_send(data->engine, data->chat_id, 
        "gpt-4", "Process this task...");
    
    archia_wait_on(data->engine, data->chat_id);
    
    // Process response
    size_t len = archia_chat_len(data->engine, data->chat_id);
    const char* result = archia_chat_message(
        data->engine, data->chat_id, len - 1);
    
    printf("Result: %s\n", result);
    return NULL;
}

int main() {
    Engine* engine = archia_new();
    pthread_t threads[4];
    ThreadData thread_data[4];
    
    // Create multiple chat sessions
    for (int i = 0; i < 4; i++) {
        thread_data[i].engine = engine;
        thread_data[i].chat_id = archia_new_chat(
            engine, "Process worker", NULL, 0, NULL);
        
        pthread_create(&threads[i], NULL, worker, &thread_data[i]);
    }
    
    // Wait for all threads
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    archia_free(engine);
    return 0;
}

Error Handling

Robust error handling example:

#include <archia.h>
#include <stdio.h>

// Helper function for safe operations
int safe_send(Engine* engine, size_t chat_id, 
              const char* model, const char* message) {
    Error* err = archia_send(engine, chat_id, model, message);
    
    if (err) {
        int code = archia_error_code(err);
        const char* msg = archia_error_message(err);
        
        switch (code) {
            case ARCHIA_ERR_INVALID_CHAT:
                fprintf(stderr, "Invalid chat ID: %s\n", msg);
                break;
            case ARCHIA_ERR_MODEL_NOT_FOUND:
                fprintf(stderr, "Model not available: %s\n", msg);
                break;
            case ARCHIA_ERR_API_KEY:
                fprintf(stderr, "API key issue: %s\n", msg);
                break;
            case ARCHIA_ERR_NETWORK:
                fprintf(stderr, "Network error: %s\n", msg);
                break;
            case ARCHIA_ERR_RATE_LIMIT:
                fprintf(stderr, "Rate limit hit: %s\n", msg);
                break;
            default:
                fprintf(stderr, "Error %d: %s\n", code, msg);
        }
        
        archia_error_free(err);
        return -1;
    }
    
    return 0;
}

int main() {
    Engine* engine = archia_new();
    if (!engine) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }
    
    size_t chat = archia_new_chat(engine, "Assistant", NULL, 0, NULL);
    
    // Try sending with error handling
    if (safe_send(engine, chat, "gpt-4", "Hello!") == 0) {
        archia_wait_on(engine, chat);
        // Process response...
    }
    
    archia_free(engine);
    return 0;
}

C++ Wrapper

For C++ projects, use our RAII wrapper for automatic resource management:

#include <archia.hpp>
#include <iostream>

int main() {
    try {
        // RAII - automatic cleanup
        archia::Engine engine;
        
        // Create chat
        auto chat = engine.newChat("You are a C++ expert");
        
        // Send message
        chat.send("claude-3-5-sonnet", 
                  "Write a modern C++20 example");
        
        // Wait and get response
        auto response = chat.waitAndGetResponse();
        std::cout << "Assistant: " << response << std::endl;
        
    } catch (const archia::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

Error Handling
C Error Handling

Error* err = archia_send(engine, chat_id, model, message);
if (err) {
    const char* msg = archia_error_message(err);
    int code = archia_error_code(err);
    
    switch (code) {
        case ARCHIA_ERR_INVALID_CHAT:
            fprintf(stderr, "Invalid chat ID\n");
            break;
        case ARCHIA_ERR_MODEL_NOT_FOUND:
            fprintf(stderr, "Model not available: %s\n", msg);
            break;
        case ARCHIA_ERR_NETWORK:
            fprintf(stderr, "Network error: %s\n", msg);
            break;
        default:
            fprintf(stderr, "Unknown error: %s\n", msg);
    }
    
    archia_error_free(err);
}

C++ Exception Handling

try {
    engine.send(chat_id, model, message);
} catch (const archia::InvalidChatError& e) {
    // Handle invalid chat
} catch (const archia::ModelError& e) {
    // Handle model issues
} catch (const archia::NetworkError& e) {
    // Handle network problems
} catch (const archia::Error& e) {
    // Generic error handling
}

Memory Management
Best Practices

    Always free the engine when done
    Check return values for NULL/error
    Don’t free message strings - they’re owned by the engine
    Free error objects after handling

// Good pattern
Engine* engine = archia_new();
if (!engine) {
    return -1;
}

// Use engine...

// Always cleanup
archia_free(engine);  // This also frees all chats

Performance Tips

    Reuse Engine Instance: Create once, use many times
    Batch Messages: Send multiple before waiting
    Use Thread Pools: For concurrent operations
    Preallocate MCP Servers: Configure all MCPs upfront
    Stream Responses: Use streaming API for large responses

Platform Notes
Linux

    Requires glibc 2.17+
    Links against libpthread automatically

macOS

    Universal binary (Intel + Apple Silicon)
    Requires macOS 10.15+

Windows

    Supports Windows 7+
    Both MSVC and MinGW toolchains
    Requires Visual C++ Redistributables

Embedded Systems

    Supports ARM, RISC-V, MIPS
    Minimum 4MB RAM
    Optional NO_STD mode available
