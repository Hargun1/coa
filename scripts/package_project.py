"""Create a portable source/results bundle, excluding environments and QA images."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def package():
    files = [ROOT / name for name in ("README.md", "requirements.txt", "pyproject.toml", ".gitignore", "start_dashboard.bat")]
    for directory in ("adaptive_memory", "configs", "docs", "examples", "paper", "scripts", "tests", "web"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.extend(ROOT / "results" / name for name in ("benchmark.json", "manifest.json", "raw.csv", "summary.csv"))
    files.extend([ROOT / "output/pdf/research_paper.pdf", ROOT / "output/verification.json"])
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    digests = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output = ROOT / "output/adaptive-memory-project.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("ARTIFACT_CHECKSUMS.json", json.dumps(digests, indent=2))
    (ROOT / "output/artifact_checksums.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")
    print(f"Packaged {len(files)} files: {output} ({output.stat().st_size:,} bytes)")
    return output


if __name__ == "__main__":
    package()
