"""The standalone upstream verifier must fail safely without TensorFlow installed."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_ader.py"


def test_source_tampering_is_rejected(tmp_path):
    spec = importlib.util.spec_from_file_location("verify_ader", SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    body = b"original source\n"
    verifier.SOURCE_HASHES = {
        "model.py": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    }
    source = tmp_path / "model.py"
    source.write_bytes(body)
    verifier.verify_source(tmp_path)
    source.write_bytes(body + b"changed\n")
    with pytest.raises(ValueError, match="Source differs"):
        verifier.verify_source(tmp_path)


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "verification.json"
    sentinel.write_text("existing evidence", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", "missing-source", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "Use a new output directory" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "existing evidence"


def test_optimized_python_cannot_skip_verification():
    result = subprocess.run(
        [sys.executable, "-O", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "verification requires assertions" in result.stderr
