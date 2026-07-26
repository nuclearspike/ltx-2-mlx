"""Runtime identity has a stable machine-readable module entrypoint."""

from __future__ import annotations

import json

from ltx_pipelines_mlx.utils import runtime_info


def test_runtime_info_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runtime_info,
        "runtime_identity",
        lambda: {"runtime_version": "0.14.20.dev1", "runtime_commit": "abc123"},
    )

    runtime_info.main()

    assert json.loads(capsys.readouterr().out) == {
        "runtime_commit": "abc123",
        "runtime_version": "0.14.20.dev1",
    }
