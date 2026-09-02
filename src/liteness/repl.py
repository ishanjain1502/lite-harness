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
        self.session = self._open_session()
        try:
            self.runtime = self._build_runtime()
        except Exception as exc:
            self._out(f"Error: {exc}")
            return 1
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
            if stripped in _QUIT_COMMANDS:
                self._dispose()
                return 0
            if stripped.startswith(":"):
                self._out(f"unknown command: {stripped}")
                continue
            self._out(f"(turns not yet implemented: {stripped!r})")

    def _dispose(self) -> None:
        if self.runtime is not None:
            dispose_runtime(self.runtime)
            self.runtime = None

