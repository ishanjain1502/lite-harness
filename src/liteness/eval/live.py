"""Live agent execution for eval run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from liteness.context import Context
from liteness.harness import create_runtime, dispose_runtime
from liteness.loop import AgentLoop, LoopConfig
from liteness.llm import LLMProvider, MockLLMProvider, MockStep, ToolCallDraft
from liteness.plugins.filesystem import FilesystemPlugin
from liteness.plugins.telemetry import TelemetryPlugin
from liteness.providers import default_model, resolve_provider
from liteness.replay import ReplayLLMProvider
from liteness.session import Session
from liteness.tools import ToolRegistry

from liteness.eval.errors import EvalCaseError
from liteness.eval.models import EvalCase, EvalSuite

if TYPE_CHECKING:
    from liteness.harness import HarnessRuntime


@dataclass
class LiveEvalConfig:
    preset: str | None = None
    project_id: str = "default"
    provider: str = "mock"
    model: str | None = None
    readme: str = "README.md"
    max_steps: int = 10
    sessions_dir: Path = Path(".sessions/evals")
    telemetry: bool = False
    clickhouse: dict | None = None
    llm: LLMProvider | None = None
    tools: ToolRegistry | None = None


def session_path_for_case(config: LiveEvalConfig, case_id: str) -> Path:
    config.sessions_dir.mkdir(parents=True, exist_ok=True)
    return config.sessions_dir / f"{case_id}.jsonl"


def execute_live_case(
    case: EvalCase,
    suite: EvalSuite,
    config: LiveEvalConfig,
) -> Path:
    """Run one live agent turn and return the durable session JSONL path."""
    if not case.input:
        raise EvalCaseError(f"case {case.id!r} has no input for live eval")

    session_path = session_path_for_case(config, case.id)
    session = Session.open(session_path)
    runtime: HarnessRuntime | None = None
    local_ctx: Context | None = None

    try:
        if config.llm is not None and config.tools is not None:
            loop = AgentLoop(
                llm=config.llm,
                tools=config.tools,
                config=LoopConfig(max_steps_per_turn=config.max_steps),
            )
        else:
            preset = config.preset or suite.preset
            if preset:
                extra_plugins: list[str] | None = None
                extra_plugin_config: dict[str, dict] | None = None
                if config.clickhouse is not None:
                    extra_plugins = ["clickhouse"]
                    extra_plugin_config = {"clickhouse": config.clickhouse}
                runtime = create_runtime(
                    preset_name=preset,
                    session=session,
                    project_id=config.project_id,
                    extra_plugins=extra_plugins,
                    extra_plugin_config=extra_plugin_config,
                )
                loop = _build_loop_from_runtime(runtime, config, session)
            else:
                loop, local_ctx = _build_default_loop(config, session)

        loop.run_turn(session, case.input)
    finally:
        if runtime is not None:
            dispose_runtime(runtime)
        elif local_ctx is not None:
            clickhouse_plugin = local_ctx.services.get("clickhouse")
            if clickhouse_plugin is not None:
                from liteness.plugins.clickhouse import ClickHousePlugin

                if isinstance(clickhouse_plugin, ClickHousePlugin):
                    clickhouse_plugin.uninstall(local_ctx)

    return session_path


def weather_mock_llm() -> MockLLMProvider:
    """Deterministic mock LLM for weather eval live tests."""
    return MockLLMProvider(
        steps=[
            MockStep(
                step=1,
                content="Let me check the weather.",
                tool_calls=[
                    ToolCallDraft(
                        call_id="w1",
                        name="weather",
                        arguments={"city": "Delhi"},
                    )
                ],
            ),
            MockStep(step=2, content="The weather in Delhi is sunny."),
        ]
    )


def _resolve_llm(config: LiveEvalConfig, session: Session) -> LLMProvider:
    if config.llm is not None:
        return config.llm
    if config.provider == "mock":
        return _mock_for_readme_task(config.readme)
    return resolve_provider(config.provider, model=config.model)


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


def _build_default_loop(config: LiveEvalConfig, session: Session) -> tuple[AgentLoop, Context]:
    ctx = Context()
    FilesystemPlugin().install(ctx, {})
    if config.telemetry:
        TelemetryPlugin().install(ctx, {})
    if config.clickhouse is not None:
        from liteness.plugins.clickhouse import ClickHousePlugin

        ClickHousePlugin().install(ctx, config.clickhouse)

    def event_sink(event) -> None:
        ctx.emit(
            "session/event",
            event.type,
            event.payload,
            session_event=event,
        )

    model = config.model
    if model is None:
        model = "mock" if config.provider == "mock" else default_model(config.provider)

    loop = AgentLoop(
        llm=_resolve_llm(config, session),
        tools=ctx.tools,
        config=LoopConfig(
            max_steps_per_turn=config.max_steps,
            allowed_tools=ctx.tools.names(),
            model=model,
        ),
        event_sink=event_sink,
    )
    return loop, ctx


def _build_loop_from_runtime(
    runtime: HarnessRuntime,
    config: LiveEvalConfig,
    session: Session,
) -> AgentLoop:
    telemetry_plugin = runtime.ctx.services.get("telemetry")
    if telemetry_plugin is None and config.telemetry:
        telemetry_plugin = TelemetryPlugin()
        telemetry_plugin.install(runtime.ctx, {})

    def event_sink(event) -> None:
        runtime.ctx.emit(
            "session/event",
            event.type,
            event.payload,
            session_event=event,
        )

    model = config.model
    if model is None:
        model = "mock" if config.provider == "mock" else default_model(config.provider)

    llm = _resolve_llm(config, session)
    if config.provider == "replay":
        llm = ReplayLLMProvider.from_session(session)

    return AgentLoop(
        llm=llm,
        tools=runtime.ctx.tools,
        config=LoopConfig(
            max_steps_per_turn=config.max_steps,
            allowed_tools=runtime.ctx.tools.names(),
            model=model,
            system_prompt=runtime.preset.system_prompt,
        ),
        event_sink=event_sink,
    )
