import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_312_is_consistent_across_deployment_targets():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock_header = (ROOT / "uv.lock").read_text().splitlines()[:4]

    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    assert project["project"]["requires-python"] == ">=3.12"
    assert 'requires-python = ">=3.12"' in lock_header
    assert (ROOT / "Dockerfile").read_text().startswith("FROM python:3.12-slim")
    assert 'python-version: "3.12"' in (ROOT / ".github/workflows/ci.yml").read_text()


def test_vercel_uses_lightweight_serverless_web_mode():
    config = json.loads((ROOT / "vercel.json").read_text())

    assert config["builds"] == [{"src": "app.py", "use": "@vercel/python"}]
    assert config["env"]["BTC_ORACLE_MODE"] == "demo"
    assert (ROOT / "uv.lock").stat().st_size < 100_000
