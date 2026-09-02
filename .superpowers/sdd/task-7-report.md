Task 7 — Per-turn error continuation
====================================

What I implemented
------------------
- Added a test `test_repl_continues_after_turn_error` to `tests/test_repl.py` which simulates an LLM error on the first turn then recovers on the second turn.
- Hardened `ReplSession._run_turn` in `src/liteness/repl.py` by wrapping the `run_turn` call with a try/except that:
  - prints a blank line when partial streaming output exists,
  - prints `[unexpected error: <msg>]` for unexpected exceptions,
  - returns (continues the REPL) instead of allowing exceptions to propagate.

TDD steps and commands run
--------------------------
1) Add the failing test (appended to `tests/test_repl.py`).
2) Run the new test:

   pytest tests/test_repl.py::test_repl_continues_after_turn_error -q

   Result: Passed (see notes below — the test passed even before the code change).

3) Implemented the try/except around `loop.run_turn` in `src/liteness/repl.py`.
4) Re-run the test:

   pytest tests/test_repl.py::test_repl_continues_after_turn_error -q

   Result: Passed.

5) Commit changes:

   git add src/liteness/repl.py tests/test_repl.py
   git commit -m "feat(repl): continue prompting after turn errors"

TDD evidence (RED / GREEN)
--------------------------
- Expected flow in brief: test would fail before code change and pass after (RED → GREEN).
- Actual result: The new test passed before making the change and continued to pass after the change.
  - Reasoning: AgentLoop/run_turn (or the mock provider) already handled the simulated LlmError path in this codebase, so the test did not fail initially.
  - Conclusion: Behavior is already resilient to the simulated error, but I implemented the explicit try/except per the task brief for clarity and defense-in-depth.
  - Final status: GREEN — tests pass and the REPL now explicitly guards against unexpected exceptions per the brief.

Files changed
-------------
- modified: src/liteness/repl.py
  - Added try/except around `loop.run_turn` to catch unexpected exceptions and continue the REPL.
- modified: tests/test_repl.py
  - Added `test_repl_continues_after_turn_error`.
- added:  .superpowers/sdd/task-7-report.md (this file)

Commits created
---------------
- d7b063e — feat(repl): continue prompting after turn errors

Self-review and concerns
------------------------
- Self-review: Changes are small and local to REPL turn execution. Lints pass for edited files.
- Concern: Catching broad Exception keeps the REPL alive (desired), but it may hide internal bugs. This is intentional per the brief (defensive REPL behavior). When Task 8 adds signal handling, we should ensure the CancelToken and SIGINT interaction does not suppress intended cancel semantics.

Report location
---------------
E:/Projects/harness/lite-ness/.superpowers/sdd/task-7-report.md

Code review fixes (2026-09-03)
================================

Fix 1 — preserve cancellation
-----------------------------
- `src/liteness/repl.py:12` imports `CancelledError` with `CancelToken`.
- `src/liteness/repl.py:143` re-raises `CancelledError` before the broad
  `except Exception`, while the existing outer `finally` still restores
  `event_sink`.
- `tests/test_repl.py:235` verifies cancellation propagates, is not printed as
  an unexpected error, and restores the original event sink.

Cancellation RED:

    pytest tests/test_repl.py::test_repl_reraises_cancelled_error -q

    F                                                                        [100%]
    FAILED tests/test_repl.py::test_repl_reraises_cancelled_error - Failed: DID NOT RAISE <class 'liteness.types.CancelledError'>
    1 failed in 0.15s

Cancellation GREEN:

    pytest tests/test_repl.py::test_repl_reraises_cancelled_error -q

    .                                                                        [100%]
    1 passed in 0.14s

Fix 2 — genuine continuation RED/GREEN
--------------------------------------
- Strengthened `tests/test_repl.py:206` so the first direct call to
  `loop.run_turn` raises `LlmError`, bypassing `AgentLoop`'s internal
  conversion of provider errors to a normal `TurnResult`. The second prompt
  calls the real loop and produces `recovered`.
- Temporarily removed the `_run_turn` broad guard and ran:

    pytest tests/test_repl.py::test_repl_continues_after_turn_error -q

    F                                                                        [100%]
    ================================== FAILURES ===================================
    ____________________ test_repl_continues_after_turn_error _____________________
    E           liteness.types.LlmError: boom
    tests\test_repl.py:223: LlmError
    =========================== short test summary info ===========================
    FAILED tests/test_repl.py::test_repl_continues_after_turn_error - liteness.ty...
    1 failed in 0.17s

- Restored the guard (including the new cancellation re-raise) and ran:

    pytest tests/test_repl.py::test_repl_continues_after_turn_error -q

    .                                                                        [100%]
    1 passed in 0.12s

Final verification
------------------

    pytest tests/test_repl.py tests/test_repl_commands.py -q

    .................                                                        [100%]
    17 passed in 0.20s

`git diff --check` completed successfully. IDE lint diagnostics reported no
errors in `src/liteness/repl.py` or `tests/test_repl.py`.

