"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from liteness.llm import MockLLMProvider, MockStep, OpenAILLMProvider, ToolCallDraft
from liteness.loop import AgentLoop, LoopConfig
from liteness.session import Session
from liteness.tools import ToolRegistry, default_registry


def _format_size(content: str) -> str:
    size = len(content.encode("utf-8"))
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _summarize_session(session: Session, result_steps: int) -> str:
    lines = [f"Session: {session.session_id}", ""]
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
    lines.append("Completed" if result_steps else "Stopped")
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


def run_command(args: argparse.Namespace) -> int:
    registry = default_registry()
    if args.provider == "openai":
        llm = OpenAILLMProvider(model=args.model)
    else:
        llm = _mock_for_readme_task(args.readme)

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        config=LoopConfig(
            max_steps_per_turn=args.max_steps,
            allowed_tools=registry.names(),
            model=args.model,
        ),
    )

    session = Session()
    result = loop.run_turn(session, args.prompt)

    print(_summarize_session(session, result.steps_run))
    if result.final_output:
        print()
        print(result.final_output)

    if args.verbose:
        print()
        print("--- session.events ---")
        for event in session.events:
            print(f"  [{event.type}] turn={event.turn} step={event.step}")

    return 0 if result.status == "completed" else 1


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
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
