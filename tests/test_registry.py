import json
import sys
from pathlib import Path
from subprocess import run

HERE = Path(__file__).parent


def test_registerable_when_not_imported():
    proc = run(
        [sys.executable, "-I", HERE / "data/check-registry.py"],
        capture_output=True,
        check=True,
    )
    results = json.loads(proc.stdout)
    assert "zarrs" not in results["imported_modules"]
    assert results["is_registered"]
