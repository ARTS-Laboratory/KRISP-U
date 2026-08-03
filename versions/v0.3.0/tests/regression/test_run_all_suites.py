from pathlib import Path

import evaluation.runners.run_all_suites as wrapper


def test_run_all_suites_preserves_order_and_records_failures(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def fake_run(config_path: Path, output_root: Path) -> Path:
        name = config_path.stem
        calls.append(name)
        if name == "canonical_2d":
            raise RuntimeError("synthetic failure")
        return output_root / name

    monkeypatch.setattr(wrapper, "run_benchmark", fake_run)
    summary = wrapper.run_all_suites(
        ("development", "canonical_2d", "canonical_doe"),
        tmp_path,
    )

    assert calls == ["development", "canonical_2d", "canonical_doe"]
    assert [row["status"] for row in summary["results"]] == [
        "completed",
        "failed",
        "completed",
    ]
    manifest = tmp_path / "all_suites_manifest.yaml"
    assert manifest.exists()
