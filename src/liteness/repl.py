"""Interactive REPL for liteness — one live session across prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from liteness.harness import HarnessRuntime, create_runtime, dispose_runtime
from liteness.loop import AgentLoop
from liteness.providers import default_model
from liteness.session import Session


@dataclass
class ReplConfig:
    preset: str | None
    provider: str
    model: str | None
    project_id: str
    max_steps: int
    session_file: str | None
    readme: str
    telemetry: bool
    verbose: bool
    # NOTE: keep ReplConfig matching the task brief exactly.
    # replay is injected into the argparse.Namespace when building the loop.


_QUIT_COMMANDS = {":q", ":quit", ":exit"}
def _format_size(content: str) -> str:
    size = len(content.encode("utf-8"))
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
class _QuitRepl(Exception):
    """Internal control-flow signal for :q commands."""

class ReplSession:
    def __init__(
        self,
        config: ReplConfig,
        *,
        input_fn: Callable[..., str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self._input = input_fn
        self._out = output_fn
        self.session: Session | None = None
        self.runtime: HarnessRuntime | None = None
        self.loop: AgentLoop | None = None

    def run(self) -> int:
        if self.session is None:
            self.session = self._open_session()
        try:
            self.runtime = self._build_runtime()
        except Exception as exc:
            self._out(f"Error: {exc}")
            return 1
        if self.loop is None:
            self.loop = self._build_loop()
        self._print_banner()
        return self._read_loop()

    def _open_session(self) -> Session:
        if self.config.session_file:
            return Session.open(self.config.session_file)
        return Session()

    def _build_runtime(self) -> HarnessRuntime | None:
        if not self.config.preset:
            return None
        return create_runtime(
            preset_name=self.config.preset,
            session=self.session,  # type: ignore[arg-type]
            project_id=self.config.project_id,
        )

    def _build_loop(self) -> AgentLoop:
        import argparse
        from liteness.cli import _build_loop as cli_build_loop

        args = argparse.Namespace(replay=False, **self.config.__dict__)
        return cli_build_loop(args, self.session, runtime=self.runtime)

    def _print_banner(self) -> None:
        model = self.config.model or default_model(self.config.provider)
        tools = self.loop.tools.names() if self.loop else []
        self._out("liteness repl")
        self._out(f"  preset:    {self.config.preset or '(none)'}")
        self._out(f"  provider: {self.config.provider} / {model}")
        self._out(f"  tools:     {', '.join(tools) if tools else '(none)'}")
        self._out(f"  session:   {self.session.session_id}")  # type: ignore[union-attr]
        self._out("  commands:  :q :tools :history :reset :preset <name> :report")

    def _read_loop(self) -> int:
        while True:
            try:
                line = self._input("you> ")
            except EOFError:
                self._dispose()
                return 0
            except KeyboardInterrupt:
                self._out("")
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(":"):
                try:
                    self._handle_command(stripped)
                except _QuitRepl:
                    return 0
                continue
            self._run_turn(stripped)

    def _run_turn(self, prompt: str) -> None:
        from liteness.types import CancelToken

        cancel = CancelToken()
        self._out(f"you> {prompt}")
        streamed: list[str] = []

        def event_sink(event) -> None:
            self._on_event(event, streamed)

        loop = self.loop
        prev_sink = None
        if loop is None:
            self._out("Error: no loop configured")
            return
        prev_sink = loop.event_sink
        try:
            loop.event_sink = event_sink
            result = loop.run_turn(self.session, prompt, cancel=cancel)  # type: ignore[arg-type]
        finally:
            loop.event_sink = prev_sink

        if streamed:
            self._out("")
        if result.final_output and result.final_output not in "".join(streamed):
            self._out(result.final_output)
        self._out(f"[turn {result.turn} · {result.status} · {result.stop_reason.value} · {result.steps_run} steps]")

    def _on_event(self, event, streamed: list[str]) -> None:
        if event.type == "assistant/chunk":
            delta = event.payload.get("content_delta") or ""
            if delta:
                self._out(delta)
                streamed.append(delta)
            for d in event.payload.get("tool_call_deltas") or []:
                name = d.get("name")
                if name:
                    self._out(f"\n-> tool: {name}")
        elif event.type == "assistant/message":
            content = event.payload.get("content", "")
            if content:
                self._out(content)
                streamed.append(content)
        elif event.type == "tool/result":
            content = event.payload.get("content", "")
            if event.payload.get("is_error"):
                self._out(f"-> error: {content[:80]}")
            else:
                self._out(f"-> result: {_format_size(content)}")

    def _dispose(self) -> None:
        if self.runtime is not None:
            dispose_runtime(self.runtime)
            self.runtime = None
    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else None

        if cmd in _QUIT_COMMANDS:
            raise _QuitRepl()

        if cmd == ":tools":
            names = self.loop.tools.names() if self.loop else []
            self._out(", ".join(names) if names else "(no tools)")
            return True

        if cmd == ":history":
            for e in self.session.events:  # type: ignore[union-attr]
                self._out(f"  [{e.type}] turn={e.turn} step={e.step}")
            return True

        if cmd == ":report":
            from liteness.telemetry.projector import format_trace_report, project_events

            self._out(format_trace_report(project_events(self.session.events)))  # type: ignore[arg-type]
            return True

        if cmd == ":reset":
            self._do_reset()
            return True

        if cmd == ":preset":
            if not arg:
                self._out("usage: :preset <name>")
                return True
            self._do_preset(arg)
            return True

        self._out(f"unknown command: {line}")
        return False

    def _do_reset(self) -> None:
        session = self.session  # type: ignore[assignment]
        session.events.clear()
        session._turn = 0
        session._step = 0
        if session.log_path is not None:
            session.log_path.parent.mkdir(parents=True, exist_ok=True)
            with session.log_path.open("w", encoding="utf-8") as handle:
                handle.truncate(0)
                handle.flush()
                import os
                os.fsync(handle.fileno())
        self._out(f"session reset (id: {session.session_id} retained)")

    def _do_preset(self, name: str) -> None:
        if self.runtime is not None:
            dispose_runtime(self.runtime)
            self.runtime = None
        try:
            self.runtime = create_runtime(
                preset_name=name,
                session=self.session,  # type: ignore[arg-type]
                project_id=self.config.project_id,
            )
        except Exception as exc:
            self._out(f"Error switching preset: {exc}")
            return
        self.loop = self._build_loop()
        tools = self.loop.tools.names() if self.loop else []
        self._out(f"switched to preset: {name} (tools: {', '.join(tools)})")
