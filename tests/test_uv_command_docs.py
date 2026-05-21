from __future__ import annotations

from pathlib import Path


UV_COMMAND_FILES = [
    Path("Makefile"),
    Path("README.md"),
    Path("docs/environment.md"),
    Path("docs/deberta_debug.md"),
    Path("docs/escalating_model_threshold_sweep.md"),
    Path("src/cli/README.md"),
]


def test_uv_active_commands_disable_project_environment():
    offenders = []
    for path in UV_COMMAND_FILES:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "uv run --active" in line and "uv run --active --no-project" not in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")

    assert not offenders, "uv active commands must include --no-project:\n" + "\n".join(
        offenders
    )
