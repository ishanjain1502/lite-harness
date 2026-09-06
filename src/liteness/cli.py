"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from liteness.env import load_dotenv
from liteness.harness import create_runtime, dispose_runtime, HarnessRuntime
from liteness.llm import MockLLMProvider, MockStep, ToolCallDraft
from liteness.providers import default_model, resolve_provider
from liteness.loop import AgentLoop, LoopConfig
from liteness.plugins import get_plugin
from liteness.plugins.base import PluginConfigError
from liteness.plugins.rag import RAGPlugin
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.replay import ReplayLLMProvider
from liteness.session import Session, find_orphan_tool_calls, recover_orphans
from liteness.telemetry.projector import format_trace_report, project_events
from liteness.repl import ReplConfig, ReplSession
from liteness.eval.baseline import compare_baseline, load_report
from liteness.eval.errors import BaselineMismatchError, EvalError
from liteness.eval.live import LiveEvalConfig
from liteness.eval.report import format_eval_report
from liteness.eval.runner import run_session_file, run_suite_file, run_suite_live


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


def _add_clickhouse_flags(
    parser: argparse.ArgumentParser,
    *,
    repl_defaults: bool = False,
) -> None:
    if repl_defaults:
        parser.add_argument(
            "--clickhouse",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Export events to ClickHouse (default: on for repl)",
        )
        parser.add_argument(
            "--clickhouse-full",
            action="store_true",
            help="Export events to ClickHouse with raw payloads",
        )
    else:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--clickhouse",
            action="store_true",
            help="Export events to ClickHouse (redacted)",
        )
        group.add_argument(
            "--clickhouse-full",
            action="store_true",
            help="Export events to ClickHouse with raw payloads",
        )
    parser.add_argument("--clickhouse-url", default=None, help="ClickHouse HTTP URL")
    parser.add_argument("--clickhouse-database", default="liteness")


def _clickhouse_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "clickhouse", False) and not getattr(args, "clickhouse_full", False):
        return None
    mode = "full" if getattr(args, "clickhouse_full", False) else "redacted"
    url = (
        getattr(args, "clickhouse_url", None)
        or os.environ.get("LITENESS_CLICKHOUSE_URL")
        or "http://localhost:8123"
    )
    return {
        "url": url,
        "database": getattr(args, "clickhouse_database", "liteness") or "liteness",
        "mode": mode,
        "user": os.environ.get("LITENESS_CLICKHOUSE_USER"),
        "password": os.environ.get("LITENESS_CLICKHOUSE_PASSWORD"),
    }


def _resolve_llm(args: argparse.Namespace, session: Session):
    if args.replay:
        return ReplayLLMProvider.from_session(session)
    if args.provider == "mock":
        return _mock_for_readme_task(args.readme)
    return resolve_provider(args.provider, model=args.model)


def _build_loop(
    args: argparse.Namespace,
    session: Session,
    runtime=None,
    local_cleanup: list | None = None,
) -> AgentLoop:
    telemetry_plugin: TelemetryPlugin | None = None
    system_prompt: str | None = None
    if runtime is not None:
        registry = runtime.ctx.tools
        system_prompt = runtime.preset.system_prompt
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
        if getattr(args, "telemetry", False):
            TelemetryPlugin().install(ctx, {})

        clickhouse_config = _clickhouse_config_from_args(args)
        if clickhouse_config is not None:
            from liteness.plugins.clickhouse import ClickHousePlugin

            clickhouse_plugin = ClickHousePlugin()
            clickhouse_plugin.install(ctx, clickhouse_config)
            if local_cleanup is not None:
                local_cleanup.append((ctx, clickhouse_plugin))

        def event_sink(event) -> None:
            ctx.emit(
                "session/event",
                event.type,
                event.payload,
                session_event=event,
            )

    model = args.model
    if model is None:
        model = "mock" if args.provider == "mock" else default_model(args.provider)

    return AgentLoop(
        llm=_resolve_llm(args, session),
        tools=registry,
        config=LoopConfig(
            max_steps_per_turn=args.max_steps,
            allowed_tools=registry.names(),
            model=model,
            system_prompt=system_prompt,
            budget=getattr(args, "budget_config", None),
        ),
        event_sink=event_sink,
    )


def run_command(args: argparse.Namespace) -> int:
    session = _resolve_session(args)
    runtime = None
    local_cleanup: list = []

    if args.preset:
        try:
            runtime = _build_runtime(args, session)
        except PluginConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        loop = _build_loop(
            args, session, runtime=runtime, local_cleanup=local_cleanup
        )
    except PluginConfigError as exc:
        if runtime is not None:
            dispose_runtime(runtime)
        else:
            for ctx, plugin in local_cleanup:
                plugin.uninstall(ctx)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        result = loop.run_turn(session, args.prompt)
    finally:
        if runtime is not None:
            dispose_runtime(runtime)
        else:
            for ctx, plugin in local_cleanup:
                plugin.uninstall(ctx)

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


def _build_runtime(args: argparse.Namespace, session: Session) -> HarnessRuntime | None:
    if not args.preset:
        return None
    extra_plugins: list[str] | None = None
    extra_plugin_config: dict[str, dict[str, Any]] | None = None
    clickhouse_config = _clickhouse_config_from_args(args)
    if clickhouse_config is not None:
        extra_plugins = ["clickhouse"]
        extra_plugin_config = {"clickhouse": clickhouse_config}
    return create_runtime(
        preset_name=args.preset,
        session=session,
        project_id=args.project_id,
        extra_plugins=extra_plugins,
        extra_plugin_config=extra_plugin_config,
    )


def repl_command(args: argparse.Namespace) -> int:
    if args.clickhouse or args.clickhouse_full:
        from liteness.clickhouse.docker import ensure_clickhouse_running

        ensure_clickhouse_running(
            url=(
                args.clickhouse_url
                or os.environ.get("LITENESS_CLICKHOUSE_URL")
                or "http://localhost:8123"
            ),
            user=os.environ.get("LITENESS_CLICKHOUSE_USER"),
            password=os.environ.get("LITENESS_CLICKHOUSE_PASSWORD"),
        )

    config = ReplConfig(
        preset=args.preset,
        provider=args.provider,
        model=args.model,
        project_id=args.project_id,
        max_steps=args.max_steps,
        session_file=args.session_file,
        readme=args.readme,
        telemetry=args.telemetry,
        verbose=args.verbose,
        debug=args.debug,
        eval_file=args.eval_file,
        clickhouse=args.clickhouse,
        clickhouse_full=args.clickhouse_full,
        clickhouse_url=args.clickhouse_url,
        clickhouse_database=args.clickhouse_database,
    )
    return ReplSession(config).run()


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


def _make_clickhouse_exporter(config: dict[str, Any]):
    from liteness.clickhouse.client import (
        HttpClickHouseClient,
        clickhouse_credentials_from_config,
    )
    from liteness.clickhouse.exporter import ClickHouseExporter
    from liteness.clickhouse.projector import ClickHouseProjector

    user, password = clickhouse_credentials_from_config(config)
    client = config.get("client") or HttpClickHouseClient(
        url=config["url"],
        database=config["database"],
        user=user,
        password=password,
    )
    try:
        client.ensure_schema()
    except Exception:
        logging.getLogger("liteness.clickhouse").warning(
            "schema ensure failed", exc_info=True
        )
    return ClickHouseExporter(
        client,
        spill_path=Path(config.get("spill_path") or ".liteness/clickhouse-spill.jsonl"),
        projector=ClickHouseProjector(mode=config.get("mode", "redacted")),
    )


def export_clickhouse_command(args: argparse.Namespace) -> int:
    config = {
        "url": args.clickhouse_url
        or os.environ.get("LITENESS_CLICKHOUSE_URL")
        or "http://localhost:8123",
        "database": args.clickhouse_database,
        "mode": "full" if args.full else "redacted",
        "user": os.environ.get("LITENESS_CLICKHOUSE_USER"),
        "password": os.environ.get("LITENESS_CLICKHOUSE_PASSWORD"),
    }
    session = Session.load_from_jsonl(args.session_file, recover=False)
    exporter = _make_clickhouse_exporter(config)
    projector = exporter.projector
    projected_count = 0
    try:
        for event in session.events:
            row = projector.project_event(event)
            if row is not None:
                exporter.emit_session(row)
                projected_count += 1
    finally:
        exporter.shutdown(session_id=session.session_id)
    print(f"Exported {projected_count} events from {args.session_file}")
    return 0


def _export_eval_report(args: argparse.Namespace, report) -> None:
    config = _clickhouse_config_from_args(args)
    if config is None:
        return
    from liteness.clickhouse.projector import project_eval_result

    exporter = _make_clickhouse_exporter(config)
    projector = exporter.projector
    preset = getattr(args, "preset", "") or ""
    provider = getattr(args, "provider", "") or ""
    model = getattr(args, "model", "") or ""
    eval_rows: list[dict[str, Any]] = []
    try:
        for case in report.cases:
            session_id = ""
            if case.session_path:
                path = Path(case.session_path)
                if path.exists():
                    session = Session.load_from_jsonl(path, recover=False)
                    session_id = session.session_id
                    for event in session.events:
                        row = projector.project_event(event)
                        if row is not None:
                            exporter.emit_session(row)
            for result in case.results:
                row = project_eval_result(
                    report=report,
                    case=case,
                    result=result,
                    session_id=session_id,
                    mode=config.get("mode", "redacted"),
                    preset=preset,
                    provider=provider,
                    model=model or "",
                )
                eval_rows.append(row.to_insert_dict())
        if eval_rows:
            # Sync insert: eval_results must land before shutdown would spill small batches.
            try:
                exporter.client.insert_rows("eval_results", eval_rows)
            except Exception:
                for row in eval_rows:
                    exporter._spill("eval_results", row)
    finally:
        exporter.shutdown()


def _finish_eval_command(args: argparse.Namespace, report) -> int:
    if args.baseline:
        try:
            compare_baseline(
                report,
                args.baseline,
                fail_on_regression=args.fail_on_regression,
            )
        except BaselineMismatchError as exc:
            print(f"baseline regression: {exc}", file=sys.stderr)
            return 1

    if args.format == "json":
        import json as json_mod

        print(json_mod.dumps(report.to_dict(), indent=2))
    else:
        print(format_eval_report(report, verbose=args.verbose))

    if args.output:
        import json as json_mod

        Path(args.output).write_text(
            json_mod.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    if _clickhouse_config_from_args(args):
        try:
            _export_eval_report(args, report)
        except Exception:
            logging.getLogger("liteness.clickhouse").warning(
                "eval ClickHouse export failed", exc_info=True
            )

    pass_rate = float(report.summary.get("pass_rate", 0.0))
    if pass_rate < args.min_score:
        return 1
    if report.summary.get("failed", 0) > 0:
        return 1
    return 0


def eval_session_command(args: argparse.Namespace) -> int:
    try:
        report = run_session_file(
            args.session_file,
            eval_file=args.eval_file,
            case_id=args.case,
            recover=args.recover,
            fail_fast=args.fail_fast,
        )
    except EvalError as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        return 1

    return _finish_eval_command(args, report)


def eval_run_command(args: argparse.Namespace) -> int:
    live_config = LiveEvalConfig(
        preset=args.preset,
        project_id=args.project_id,
        provider=args.provider,
        model=args.model,
        readme=args.readme,
        max_steps=args.max_steps,
        sessions_dir=Path(args.sessions_dir),
        telemetry=args.telemetry,
        clickhouse=_clickhouse_config_from_args(args),
    )
    try:
        report = run_suite_live(
            args.eval_file,
            case_id=args.case,
            live_config=live_config,
            recover=args.recover,
            fail_fast=args.fail_fast,
        )
    except EvalError as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        return 1

    return _finish_eval_command(args, report)


def eval_compare_command(args: argparse.Namespace) -> int:
    try:
        report = load_report(args.report)
        result = compare_baseline(
            report,
            args.baseline,
            fail_on_regression=args.fail_on_regression,
        )
    except EvalError as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        return 1
    except BaselineMismatchError as exc:
        print(f"baseline regression: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        import json as json_mod

        print(json_mod.dumps(result, indent=2))
    else:
        status = "PASS" if result["matched"] else "FAIL"
        print(f"Baseline compare: {status}")
        if result["issues"]:
            print("Issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")
    return 0 if result["matched"] else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="liteness", description="lite-ness harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one turn")
    run_parser.add_argument("prompt", help="User task prompt")
    run_parser.add_argument(
        "--preset",
        help="Built-in agent preset (e.g. researcher, coder, video_editor)",
    )
    run_parser.add_argument(
        "--project-id",
        default="default",
        help="Project namespace for memory (default: default)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["mock", "openai", "google", "commandcode"],
        default="mock",
        help="LLM provider (default: mock)",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Model id (provider default if omitted)",
    )
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
    _add_clickhouse_flags(run_parser)
    run_parser.set_defaults(func=run_command)

    repl_parser = sub.add_parser("repl", help="Interactive REPL — one live session across prompts")
    repl_parser.add_argument("--preset", help="Built-in agent preset (e.g. researcher, coder, video_editor)")
    repl_parser.add_argument("--project-id", default="default", help="Project namespace for memory (default: default)")
    repl_parser.add_argument(
        "--provider",
        choices=["mock", "openai", "google", "commandcode"],
        default="mock",
        help="LLM provider (default: mock)",
    )
    repl_parser.add_argument("--model", default=None, help="Model id (provider default if omitted)")
    repl_parser.add_argument("--readme", default="README.md", help="README path for mock demo (default: README.md)")
    repl_parser.add_argument("--max-steps", type=int, default=10)
    repl_parser.add_argument("--session-file", metavar="PATH", default=None)
    repl_parser.add_argument(
        "--eval-file",
        help="Default eval suite YAML for repl eval command",
    )
    repl_parser.add_argument("--telemetry", action="store_true")
    repl_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show tool calls, result sizes, streaming chunks, and turn summary",
    )
    repl_parser.add_argument("-v", "--verbose", action="store_true")
    _add_clickhouse_flags(repl_parser, repl_defaults=True)
    repl_parser.set_defaults(func=repl_command)

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

    export_ch_parser = sub.add_parser(
        "export-clickhouse", help="Backfill session JSONL into ClickHouse"
    )
    export_ch_parser.add_argument("session_file", help="JSONL session log path")
    export_ch_parser.add_argument(
        "--full",
        action="store_true",
        help="Export with raw payloads (default: redacted)",
    )
    export_ch_parser.add_argument("--clickhouse-url", default=None, help="ClickHouse HTTP URL")
    export_ch_parser.add_argument("--clickhouse-database", default="liteness")
    export_ch_parser.set_defaults(func=export_clickhouse_command)

    eval_parser = sub.add_parser("eval", help="Evaluate recorded agent sessions")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_session_parser = eval_sub.add_parser(
        "session", help="Evaluate an existing recorded session JSONL"
    )
    eval_session_parser.add_argument("session_file", help="JSONL session log path")
    eval_session_parser.add_argument(
        "--eval-file",
        help="Eval suite YAML (optional; uses default evaluators if omitted)",
    )
    eval_session_parser.add_argument("--case", help="Case id when using --eval-file")
    eval_session_parser.add_argument(
        "--format",
        choices=["cli", "json"],
        default="cli",
        help="Output format (default: cli)",
    )
    eval_session_parser.add_argument("--baseline", help="Baseline JSON for regression compare")
    eval_session_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum suite pass rate required (default: 0)",
    )
    eval_session_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when baseline comparison regresses",
    )
    eval_session_parser.add_argument("--output", help="Write EvalReport JSON to path")
    eval_session_parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover orphan tool/call events before evaluation",
    )
    eval_session_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first evaluator exception",
    )
    eval_session_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-evaluator details",
    )
    _add_clickhouse_flags(eval_session_parser)
    eval_session_parser.set_defaults(func=eval_session_command)

    eval_run_parser = eval_sub.add_parser(
        "run", help="Run live agent cases from an eval suite YAML"
    )
    eval_run_parser.add_argument("eval_file", help="Eval suite YAML path")
    eval_run_parser.add_argument("--case", help="Run a single case id")
    eval_run_parser.add_argument("--preset", help="Override suite preset for live runs")
    eval_run_parser.add_argument(
        "--project-id",
        default="default",
        help="Project namespace for memory (default: default)",
    )
    eval_run_parser.add_argument(
        "--provider",
        choices=["mock", "openai", "google", "commandcode"],
        default="mock",
        help="LLM provider for live runs (default: mock)",
    )
    eval_run_parser.add_argument("--model", default=None, help="Model id (provider default if omitted)")
    eval_run_parser.add_argument(
        "--readme",
        default="README.md",
        help="README path for mock demo (default: README.md)",
    )
    eval_run_parser.add_argument("--max-steps", type=int, default=10)
    eval_run_parser.add_argument(
        "--sessions-dir",
        default=".sessions/evals",
        help="Directory for recorded eval session JSONL files",
    )
    eval_run_parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable telemetry plugin during live runs",
    )
    eval_run_parser.add_argument(
        "--format",
        choices=["cli", "json"],
        default="cli",
        help="Output format (default: cli)",
    )
    eval_run_parser.add_argument("--baseline", help="Baseline JSON for regression compare")
    eval_run_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum suite pass rate required (default: 0)",
    )
    eval_run_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when baseline comparison regresses",
    )
    eval_run_parser.add_argument("--output", help="Write EvalReport JSON to path")
    eval_run_parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover orphan tool/call events before evaluation",
    )
    eval_run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first evaluator exception",
    )
    eval_run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-evaluator details",
    )
    _add_clickhouse_flags(eval_run_parser)
    eval_run_parser.set_defaults(func=eval_run_command)

    eval_compare_parser = eval_sub.add_parser(
        "compare", help="Compare an EvalReport JSON file against a baseline"
    )
    eval_compare_parser.add_argument("report", help="EvalReport JSON path")
    eval_compare_parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    eval_compare_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when baseline comparison regresses",
    )
    eval_compare_parser.add_argument(
        "--format",
        choices=["cli", "json"],
        default="cli",
        help="Output format (default: cli)",
    )
    eval_compare_parser.set_defaults(func=eval_compare_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
