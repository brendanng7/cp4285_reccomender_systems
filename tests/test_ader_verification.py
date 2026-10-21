"""The standalone upstream verifier must fail safely without TensorFlow installed."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_ader.py"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_smoke_determinism_change_is_gpu_only_and_recorded(tmp_path, monkeypatch, device):
    spec = importlib.util.spec_from_file_location("verify_ader", SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    source = tmp_path / "source"
    source.mkdir()
    for name in ("ADER.py", "EWC.py", "modules.py", "util.py", "LICENSE"):
        (source / name).write_text("fixture", encoding="utf-8")
    original = (
        "exemplar_subseq = np.array(exemplar_data_logits)[:, 0].tolist()\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)\n"
        "os.environ['TF_DETERMINISTIC_OPS'] = '1'\n"
        "config = tf.ConfigProto()\n"
    )
    (source / "main.py").write_text(original, encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned-by-slurm")

    def inspect_launch(command, **kwargs):
        patched = (kwargs["cwd"] / "main.py").read_text(encoding="utf-8")
        expected = "0" if device == "gpu" else "1"
        assert f"os.environ['TF_DETERMINISTIC_OPS'] = '{expected}'" in patched
        assert kwargs["env"]["TF_DETERMINISTIC_OPS"] == expected
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-assigned-by-slurm"
        assert (source / "main.py").read_text(encoding="utf-8") == original
        patch = (output / "main-numpy-compat.patch").read_text(encoding="utf-8")
        if device == "gpu":
            assert "+os.environ['TF_DETERMINISTIC_OPS'] = '0'" in patch
        raise RuntimeError("launch inspected without TensorFlow")

    monkeypatch.setattr(verifier.subprocess, "run", inspect_launch)
    with pytest.raises(RuntimeError, match="launch inspected"):
        verifier.check_main(source, output, device)


@pytest.mark.parametrize("actual", ["GPU", "CPU", None])
def test_gpu_scoring_requires_execution_evidence(actual):
    spec = importlib.util.spec_from_file_location("verify_ader", SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    nodes = (
        []
        if actual is None
        else [SimpleNamespace(name="score", device=f"/job:localhost/device:{actual}:0")]
    )
    graphs = [SimpleNamespace(node=nodes)]
    if actual == "GPU":
        assert verifier.check_scoring_device(graphs, "score", "gpu") == [nodes[0].device]
    else:
        with pytest.raises(RuntimeError, match="did not run on gpu"):
            verifier.check_scoring_device(graphs, "score", "gpu")


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
