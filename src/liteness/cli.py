"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from liteness.llm import MockLLMProvider, MockStep, OpenAILLMProvider, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.replay import ReplayLLMProvider
from liteness.session import Session, find_orphan_tool_calls, recover_orphans
from liteness.tools import ToolRegistry, default_registry


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


def run_command(args: argparse.Namespace) -> int:
    registry = default_registry()
    session = _resolve_session(args)
    llm = _resolve_llm(args, session)

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        config=LoopConfig(
            max_steps_per_turn=args.max_steps,
            allowed_tools=registry.names(),
            model=args.model,
        ),
    )

    result = loop.run_turn(session, args.prompt)

    print(_summarize_session(session, result))
    if result.final_output:
        print()
        print(result.final_output)

    if args.verbose:
        print()
        print("--- session.events ---")
        for event in session.events:
            print(f"  [{event.type}] turn={event.turn} step={event.step}")

    return 0 if result.status == "completed" else 1


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
    run_parser.set_defaults(func=run_command)

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
