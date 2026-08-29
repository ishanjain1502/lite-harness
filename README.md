# lite-ness

A minimal agent harness: turn/step loop, streaming LLM, tool execution, and an append-only session log.

## Install

```bash
cd lite-ness
pip install -e ".[dev]"
```

## Run (mock LLM, no API key)

```bash
liteness run "Read README.md and tell me what this repository does."
```

## Run (OpenAI-compatible API)

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
liteness run "Read README.md and tell me what this repository does." --provider openai
```

## Architecture

See `discussionDocs/` for design notes (`init.md`, `day_02.md`, `day_03.md`, `day_04.md`).

Day 4 spine:

```
AgentLoop → LLMProvider.stream → Agent.decide → ToolRegistry → Session.events
```

## Tests

```bash
pytest
```
