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

See `discussionDocs/` for design notes (`init.md` through `day_08.md`).

Day 8 adds observability (traces, metrics, CLI report) and control-plane policies (budget, retry):

```
AgentLoop (control)          TelemetryPlugin (observe)
├── BudgetManager            ├── TraceStore
├── RetryPolicy              ├── MetricsStore
└── CancelToken              └── CLI report
```

Day 7 adds plugin composition, presets, memory, and RAG:

```
Preset → Plugins → Context (tools, events, effect)
                         ↓
              AgentLoop → LLM → Session events
```

### Presets

```bash
# Coder: filesystem + terminal
liteness run --preset coder "Read README.md and summarize."

# Researcher: memory + RAG (index docs first)
liteness index discussionDocs/
liteness run --preset researcher "What does day_06 say about session recovery?"
```

## Tests

```bash
pytest
```

### Durable sessions (Day 6)

```bash
# Run with JSONL log (created automatically, fsync per event)
liteness run "Read README.md and summarize." --session-file ./sessions/demo.jsonl

# Resume a session for another turn
liteness run "Now explain the architecture." --session-file ./sessions/demo.jsonl

# Recover orphan tool/call entries after a crash
liteness recover ./sessions/demo.jsonl

# Replay recorded LLM outputs from the log (no live model)
liteness run "continue" --session-file ./sessions/demo.jsonl --replay
```

### Observability (Day 8)

```bash
# Print trace tree + metrics after a run
liteness run "Read README.md and summarize." --session-file ./sessions/demo.jsonl --report

# Build report from an existing session log
liteness report ./sessions/demo.jsonl
```
