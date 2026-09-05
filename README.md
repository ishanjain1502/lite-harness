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

## REPL (interactive)

Keep one live session across prompts, with streaming output and control commands:

```bash
# mock provider, no API key
liteness repl --provider mock --readme README.md

# live provider with a preset
liteness repl --preset coder --provider openai --model gpt-4o-mini

# durable session (resumable by pointing at the same file)
liteness repl --preset researcher --provider google --session-file ./sessions/repl.jsonl
```

In the REPL:

- type a prompt and press Enter to run a turn
- `:tools` — list installed tools
- `:history` — print session events so far
- `:report` — print telemetry trace for the current session
- `:reset` — clear the current session in place
- `:preset <name>` — switch preset in place (same session)
- `:q` / Ctrl+D — quit

Ctrl+C cancels the in-flight turn and returns to the prompt.

> **Streaming note:** assistant chunks are emitted after each turn's LLM stream
> completes (buffered), not token-by-token as the stream progresses. Live
> token streaming is tracked as a follow-up — it requires emitting
> `assistant/chunk` events during iteration inside the agent loop.

## Run (OpenAI-compatible API)

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
liteness run "Read README.md and tell me what this repository does." --provider openai
```

## Run (Google Gemini)

```bash
pip install -e ".[google]"
export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
liteness run "Read README.md and summarize." --provider google --model gemini-2.0-flash
```

## Run (Command Code Provider API)

Command Code routes Anthropic-shaped models to `/messages`; this harness uses `/chat/completions` only. Pick OpenAI or open-source model IDs from [Command Code's model list](https://docs.commandcode.ai).

```bash
pip install -e ".[openai]"   # Command Code uses the OpenAI-compatible adapter
export COMMANDCODE_API_KEY=...
liteness run "List files in this directory." --provider commandcode --model gpt-4o-mini
```

Install all providers at once:

```bash
pip install -e ".[all-providers]"
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

# Video editor: ffmpeg tools + Gemini vision for natural-language edits
# Requires ffmpeg on PATH and GOOGLE_API_KEY for scene analysis
liteness run --preset video_editor --provider google --max-steps 15 \
  "Remove the intro from my-video.mp4"
```

Output videos are written to `./video-output/` by default.

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
