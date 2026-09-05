# Day 10 soak checklist

After the vertical slice works:

1. Start local ClickHouse.
2. Run real agents (`run` / `repl`, real providers, presets).
3. Run `eval session` on those JSONL files and `eval run` live suites.
4. Query: event counts vs JSONL; one failed tool; one failed eval case joined to its session.
5. Abuse: missing file, corrupt JSONL, `--recover`, ClickHouse killed mid-run, `--replay`, eval on incomplete sessions, unknown plugin, empty suite.
6. Log every problem in `discussionDocs/day_10_problems.md` (crash, warehouse mismatch, eval lie, CLI footgun).
7. Fix highest-value first: process crash / JSONL loss → wrong warehouse row → eval lie → CLI footgun → polish.
8. Repeat until run → CH → eval → query is boring.

Log issues in `discussionDocs/day_10_problems.md`.
