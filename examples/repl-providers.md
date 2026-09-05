# REPL — providers, presets, and start commands

Interactive mode keeps **one session** across prompts. Each line you type runs one agent turn with streaming output.

```bash
cd lite-ness
pip install -e ".[dev]"
```

Install provider extras as needed (or all at once):

```bash
pip install -e ".[openai]"          # OpenAI + Command Code
pip install -e ".[google]"          # Gemini
pip install -e ".[all-providers]"   # everything
pip install -e ".[clickhouse]"      # optional warehouse export
```

---

## API keys via `.env`

Providers read standard environment variables. **liteness loads a `.env` file automatically** when you run any CLI command from the project directory.

```bash
cp .env.example .env
# edit .env and paste your keys
```

Example `.env`:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
COMMANDCODE_API_KEY=...
# OPENAI_BASE_URL=https://api.openai.com/v1
# LITENESS_CLICKHOUSE_URL=http://localhost:8123
```

Then start REPL without exporting keys in the shell:

```bash
liteness repl --provider openai
liteness repl --provider google
liteness repl --provider commandcode
```

**Rules:**

| Behavior | Detail |
|----------|--------|
| Lookup order | Shell `export` wins; `.env` only fills **missing** vars |
| File location | `./.env` in the current working directory (run from `lite-ness/`) |
| Custom path | `LITENESS_ENV_FILE=/path/to/keys.env liteness repl ...` |
| Git safety | `.env` is gitignored; commit `.env.example` only |

Variable names per provider:

| Provider | Env vars |
|----------|----------|
| `openai` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `commandcode` | `COMMANDCODE_API_KEY` |
| ClickHouse | `LITENESS_CLICKHOUSE_URL` (optional; defaults to `http://localhost:8123`) |

Manual alternative (no `.env` file):

```bash
export OPENAI_API_KEY=sk-...
liteness repl --provider openai
```

---

## Quick reference — start commands

All commands assume you are in the `lite-ness/` project root.

### Mock (no API key)

Deterministic demo LLM that reads a README via `read_file`. Best for local testing.

```bash
liteness repl --provider mock --readme README.md
```

Custom README path:

```bash
liteness repl --provider mock --readme path/to/README.md
```

With durable session log:

```bash
liteness repl --provider mock --readme README.md --session-file .sessions/repl-mock.jsonl
```

### OpenAI

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...

# default model: gpt-4o-mini
liteness repl --provider openai

# explicit model
liteness repl --provider openai --model gpt-4o

# custom base URL (Azure, local proxy, etc.)
export OPENAI_BASE_URL=https://api.openai.com/v1
liteness repl --provider openai --model gpt-4o-mini
```

### Google Gemini

CLI flag is `--provider google` (not `gemini`).

```bash
pip install -e ".[google]"
export GOOGLE_API_KEY=...    # or GEMINI_API_KEY

# default model: gemini-2.0-flash
liteness repl --provider google

# explicit model
liteness repl --provider google --model gemini-2.0-flash
```

### Command Code

OpenAI-compatible Provider API. Use OpenAI or open-source model IDs from [Command Code docs](https://docs.commandcode.ai).

```bash
pip install -e ".[openai]"
export COMMANDCODE_API_KEY=...

# default model: deepseek/deepseek-v4-flash
liteness repl --provider commandcode

# OpenAI model via Command Code
liteness repl --provider commandcode --model gpt-4o-mini

# another routed model
liteness repl --provider commandcode --model deepseek/deepseek-v4-flash
```

> Command Code routes some vendors to `/messages`; this harness only supports `/chat/completions` models.

---

## Presets (tools + plugins)

Without `--preset`, REPL gets **filesystem tools only**. With a preset, you get that preset's plugins.

| Preset        | Plugins              | Typical use                          |
|---------------|----------------------|--------------------------------------|
| *(none)*      | filesystem           | quick file reads                     |
| `coder`       | filesystem, terminal | shell + files                        |
| `researcher`  | memory, rag          | notes + doc search (`--project-id`)  |
| `video_editor`| video pipeline       | needs ffmpeg + often `--provider google` |

Examples:

```bash
# coder + OpenAI
liteness repl --preset coder --provider openai --model gpt-4o-mini

# researcher + Gemini + durable session
liteness repl --preset researcher --provider google \
  --project-id my-project \
  --session-file .sessions/researcher.jsonl

# video editor (vision uses Gemini in preset config)
liteness repl --preset video_editor --provider google --max-steps 15
```

Switch preset **inside** the REPL (same session id retained):

```
you> :preset coder
you> List files in the current directory.
```

---

## Useful REPL flags

| Flag | Purpose |
|------|---------|
| `--session-file PATH` | Append-only JSONL log; resume later with same path |
| `--max-steps N` | Max tool/LLM steps per turn (default 10) |
| `--project-id ID` | Namespace for memory plugin (researcher preset) |
| `--eval-file PATH` | Default eval suite for the `eval` command in REPL |
| `--telemetry` | Live telemetry plugin + richer `:report` |
| `--debug` | Show tool calls, result sizes, streaming text, turn summary |
| `-v` / `--verbose` | Verbose eval report output (not REPL tool trace) |
| `--clickhouse` | Export events to ClickHouse (redacted) |
| `--clickhouse-full` | Export raw payloads (local warehouse only) |

ClickHouse example:

```bash
docker compose -f docker/clickhouse/docker-compose.yml up -d
liteness repl --provider mock --readme README.md --clickhouse \
  --session-file .sessions/repl-ch.jsonl
```

---

## Inside the REPL

| Input | Action |
|-------|--------|
| plain text | run one agent turn |
| `:tools` | list available tools |
| `:history` | print session events |
| `:report` | telemetry trace for current session |
| `:reset` | clear session in place (keeps session id) |
| `:preset <name>` | switch preset (e.g. `:preset researcher`) |
| `eval` / `eval --case <id>` | run evaluators on current session |
| `:q` / `:quit` / `:exit` / Ctrl+D | quit |
| Ctrl+C | cancel in-flight turn, return to prompt |

**Default output:** only the final answer after each prompt. Pass `--debug` at startup to see `-> tool:`, `-> result:`, intermediate model text, and `[turn · status · steps]` lines. Use `:history` or `:report` anytime for full session detail.

Example session:

```
you> Read README.md and summarize in two sentences.
you> :tools
you> :history
you> :report
you> :q
```

---

## Provider defaults

| `--provider`   | Default model              | Required env (or set in `.env`) |
|----------------|----------------------------|-----------------------------------|
| `mock`         | `mock`                     | none                              |
| `openai`       | `gpt-4o-mini`              | `OPENAI_API_KEY`                  |
| `google`       | `gemini-2.0-flash`         | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `commandcode`  | `deepseek/deepseek-v4-flash` | `COMMANDCODE_API_KEY`           |

Override any default with `--model <id>`. See `.env.example` for a copy-paste template.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | `export OPENAI_API_KEY=...` |
| `OpenAI provider requires httpx` | `pip install -e ".[openai]"` |
| Google import / client errors | `pip install -e ".[google]"` and set `GOOGLE_API_KEY` |
| `Unknown provider` | Use exactly: `mock`, `openai`, `google`, `commandcode` |
| Preset not found | Names: `coder`, `researcher`, `video_editor` |
| RAG empty | Run `liteness index docs/` before using `researcher` |

---

## Copy-paste smoke test (mock)

```bash
liteness repl --provider mock --readme README.md --session-file .sessions/smoke.jsonl
```

At the prompt:

```
Read README.md and tell me the project name.
:history
:q
```

Session log: `.sessions/smoke.jsonl`
