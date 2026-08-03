"""Regression checks for v0.3.0 package and output boundaries."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest
import yaml

import krispu
from evaluation.runners.config import load_config


def test_every_krispu_module_imports_without_evaluation_dependencies() -> None:
    modules = [krispu]
    modules.extend(
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(krispu.__path__, krispu.__name__ + ".")
    )
    forbidden = {"evaluation", "benchmarks", "scratch"}
    for module in modules:
        source_path = Path(module.__file__ or "")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported.isdisjoint(forbidden), source_path


def test_unknown_yaml_keys_are_rejected(tmp_path: Path) -> None:
    profile = {
        "experiment_name": "bad",
        "fields": ["smooth"],
        "methods": ["support_adjusted_krispu"],
        "final_budget": 6,
        "candidate_count": 16,
        "evaluation_grid_size": 8,
        "trials": 1,
        "initial_design": "interior_maximin",
        "initial_sample_count": 5,
        "initial_boundary_margin": 0.05,
        "bogus": True,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown YAML keys"):
        load_config(path)
