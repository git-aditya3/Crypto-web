"""Build a deterministic, hosting-ready Team-A deployment ZIP."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "team-a-web-deploy.zip"
FILES = (
    ".env.example",
    ".dockerignore",
    ".python-version",
    "app.py",
    "compose.yaml",
    "DEPLOYMENT.md",
    "Dockerfile",
    "LICENSE",
    "Procfile",
    "pyproject.toml",
    "README.md",
    "render.yaml",
    "uv.lock",
    "vercel.json",
    "requirements.txt",
    "requirements-ml.txt",
)
DIRECTORIES = ("oracle", "static", "templates")


def included_files() -> list[Path]:
    paths = [ROOT / name for name in FILES]
    for directory in DIRECTORIES:
        paths.extend(
            path for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing package files: {', '.join(missing)}")
    return sorted(paths, key=lambda path: path.as_posix())


def build() -> Path:
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in included_files():
            archive.write(path, Path("team-a-web") / path.relative_to(ROOT))
    return OUTPUT


if __name__ == "__main__":
    package = build()
    print(f"Created {package.name} ({package.stat().st_size:,} bytes)")
