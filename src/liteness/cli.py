"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from liteness.harness import create_runtime, dispose_runtime
from liteness.llm import MockLLMProvider, MockStep, OpenAILLMProvider, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.plugins import get_plugin
from liteness.plugins.base import PluginConfigError
from liteness.plugins.rag import RAGPlugin
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.replay import ReplayLLMProvider
from liteness.session import Session, find_orphan_tool_calls, recover_orphans
from liteness.telemetry.projector import format_trace_report, project_events


def _format_size(content: str) -> str:
    size = len(content.encode("utf-8"))
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _summarize_session(session: Session, result) -> str:
    lines = [f"Session: {session.session_id}", ""]
    if session.log_path is not None:
        lines.insert(1, f"Log: {session.log_path}")
        lines.insert(2, "")
    turn = session._turn
    lines.append(f"Turn {turn}")

    current_step: int | None = None
    for event in session.events:
        if event.type == "step/start":
            current_step = event.payload.get("step")
            lines.append(f"  Step {current_step}")
            lines.append("    -> LLM")
        elif event.type == "tool/call" and event.step == current_step:
            lines.append(f"    -> tool: {event.payload['name']}")
        elif event.type == "tool/result" and event.step == current_step:
            content = event.payload.get("content", "")
            if event.payload.get("is_error"):
                lines.append(f"    -> error: {content[:80]}")
            else:
                lines.append(f"    -> result: {_format_size(content)}")
        elif event.type == "assistant/message" and event.step == current_step:
            if not any(
                e.type == "tool/call" and e.step == current_step for e in session.events
            ):
                preview = event.payload.get("content", "")[:60]
                if preview:
                    lines.append(f"    -> final answer: {preview}...")

    lines.append("")
    status_label = result.status.capitalize()
    if hasattr(result, "stop_reason"):
        lines.append(f"{status_label} ({result.stop_reason.value})")
    else:
        lines.append(status_label)
    return "\n".join(lines)


def _mock_for_readme_task(readme_path: str = "README.md") -> MockLLMProvider:
    return MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                content="I'll read the README file.",
                tool_calls=[
                    ToolCallDraft(
                        call_id="call_readme",
                        name="read_file",
                        arguments={"path": readme_path},
                    )
                ],
            ),
            MockStep(
                step=2,
                content=(
                    "This repository is lite-ness, a minimal agent harness "
                    "with a turn/step loop, streaming LLM, tools, and session log."
                ),
            ),
        ]
    )


def _resolve_session(args: argparse.Namespace) -> Session:
    if args.session_file:
        return Session.open(args.session_file, recover=args.recover)
    return Session()


def _resolve_llm(args: argparse.Namespace, session: Session):
    if args.replay:
        return ReplayLLMProvider.from_session(session)
    if args.provider == "openai":
        return OpenAILLMProvider(model=args.model)
    return _mock_for_readme_task(args.readme)


def _build_loop(
    args: argparse.Namespace,
    session: Session,
    runtime=None,
) -> AgentLoop:
    telemetry_plugin: TelemetryPlugin | None = None
    if runtime is not None:
        registry = runtime.ctx.tools
        telemetry_plugin = runtime.ctx.services.get("telemetry")
        if telemetry_plugin is None and getattr(args, "telemetry", False):
            telemetry_plugin = TelemetryPlugin()
            telemetry_plugin.install(runtime.ctx, {})

        def event_sink(event) -> None:
            runtime.ctx.emit(
                "session/event",
                event.type,
                event.payload,
                session_event=event,
            )

    else:
        from liteness.context import Context
        from liteness.plugins.filesystem import FilesystemPlugin

        ctx = Context()
        FilesystemPlugin().install(ctx, {})
        registry = ctx.tools
        telemetry_plugin = None
        if getattr(args, "telemetry", False):
            telemetry_plugin = TelemetryPlugin()
            telemetry_plugin.install(ctx, {})

        def event_sink(event) -> None:
            if telemetry_plugin is not None:
                telemetry_plugin.projector.process(event)

    return AgentLoop(
        llm=_resolve_llm(args, session),
        tools=registry,
        config=LoopConfig(
            max_steps_per_turn=args.max_steps,
            allowed_tools=registry.names(),
            model=args.model,
            budget=getattr(args, "budget_config", None),
        ),
        event_sink=event_sink,
    )


def run_command(args: argparse.Namespace) -> int:
    session = _resolve_session(args)
    runtime = None

    if args.preset:
        try:
            runtime = create_runtime(
                preset_name=args.preset,
                session=session,
                project_id=args.project_id,
            )
        except PluginConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    loop = _build_loop(args, session, runtime=runtime)

    try:
        result = loop.run_turn(session, args.prompt)
    finally:
        if runtime is not None:
            dispose_runtime(runtime)

    print(_summarize_session(session, result))
    if result.final_output:
        print()
        print(result.final_output)

    if getattr(args, "report", False) or getattr(args, "telemetry", False):
        print()
        print("--- telemetry ---")
        if runtime is not None:
            plugin = runtime.ctx.services.get("telemetry")
            if plugin is not None:
                print(plugin.format_report(session.session_id))
            else:
                print(format_trace_report(project_events(session.events)))
        else:
            print(format_trace_report(project_events(session.events)))

    if args.verbose:
        print()
        print("--- session.events ---")
        for event in session.events:
            print(f"  [{event.type}] turn={event.turn} step={event.step}")

    return 0 if result.status == "completed" else 1


def index_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    index_path = Path(args.index_file)

    plugin = get_plugin("rag")
    if not isinstance(plugin, RAGPlugin):
        print("Error: rag plugin is not available", file=sys.stderr)
        return 1

    try:
        count = plugin.ingest_path(path)
        plugin.save_index(index_path)
    except PluginConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {count} chunk(s) from {path}")
    print(f"Wrote index to {index_path}")
    return 0


def recover_command(args: argparse.Namespace) -> int:
    session = Session.load_from_jsonl(args.session_file, recover=False)
    orphans = find_orphan_tool_calls(session.events)
    if not orphans:
        print(f"No orphan tool calls in {args.session_file}")
        return 0

    print(f"Found {len(orphans)} orphan tool call(s):")
    for orphan in orphans:
        print(f"  - {orphan.call_id} ({orphan.name}) turn={orphan.turn} step={orphan.step}")

    recovered = recover_orphans(session)
    print(f"Recovered {len(recovered)} tool/result event(s)")
    return 0


def report_command(args: argparse.Namespace) -> int:
    session = Session.load_from_jsonl(args.session_file, recover=args.recover)
    report = project_events(session.events)
    print(format_trace_report(report))
    if args.json:
        import json as json_mod

        trace = report.trace
        payload = {
            "trace_id": trace.trace_id,
            "outcome": trace.outcome,
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "kind": s.kind,
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                }
                for s in trace.spans
            ],
            "metrics": {
                "llm_requests": report.metrics.llm_requests,
                "tool_calls": report.metrics.tool_calls,
                "llm_p95_ms": report.metrics.llm_p95_ms,
                "total_tokens": report.metrics.total_tokens,
                "total_cost_usd": report.metrics.total_cost_usd,
            },
        }
        print(json_mod.dumps(payload, indent=2))
    return 0


def export_command(args: argparse.Namespace) -> int:
    session = Session.load_from_jsonl(args.session_file, recover=False)
    session.export_jsonl(args.output)
    print(f"Exported {len(session.events)} events to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="liteness", description="lite-ness harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one turn")
    run_parser.add_argument("prompt", help="User task prompt")
    run_parser.add_argument(
        "--preset",
        help="Built-in agent preset (e.g. researcher, coder)",
    )
    run_parser.add_argument(
        "--project-id",
        default="default",
        help="Project namespace for memory (default: default)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["mock", "openai"],
        default="mock",
        help="LLM provider (default: mock)",
    )
    run_parser.add_argument("--model", default="gpt-4o-mini", help="Model id")
    run_parser.add_argument(
        "--readme",
        default="README.md",
        help="README path for mock demo (default: README.md)",
    )
    run_parser.add_argument("--max-steps", type=int, default=10)
    run_parser.add_argument(
        "--session-file",
        metavar="PATH",
        help="JSONL session log path (created if missing; appended on each event)",
    )
    run_parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover orphan tool/call events when opening --session-file",
    )
    run_parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay assistant/message from session log instead of live LLM",
    )
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.add_argument(
        "--report",
        action="store_true",
        help="Print telemetry trace report after the turn",
    )
    run_parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable live telemetry plugin during the run",
    )
    run_parser.set_defaults(func=run_command)

    report_parser = sub.add_parser("report", help="Build telemetry report from session JSONL")
    report_parser.add_argument("session_file", help="JSONL session log path")
    report_parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover orphan tool/call events before projecting",
    )
    report_parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit machine-readable JSON summary",
    )
    report_parser.set_defaults(func=report_command)

    index_parser = sub.add_parser("index", help="Build RAG index from markdown docs")
    index_parser.add_argument("path", help="Markdown file or directory to index")
    index_parser.add_argument(
        "--index-file",
        default=".liteness/rag-index.json",
        help="Where to write the RAG index (default: .liteness/rag-index.json)",
    )
    index_parser.set_defaults(func=index_command)

    recover_parser = sub.add_parser(
        "recover", help="Synthesize tool/result for orphan tool/call events"
    )
    recover_parser.add_argument("session_file", help="JSONL session log path")
    recover_parser.set_defaults(func=recover_command)

    export_parser = sub.add_parser("export", help="Copy session log to another path")
    export_parser.add_argument("session_file", help="Source JSONL session log")
    export_parser.add_argument("output", help="Destination JSONL path")
    export_parser.set_defaults(func=export_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
