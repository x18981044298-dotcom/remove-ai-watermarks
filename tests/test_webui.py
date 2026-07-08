"""Tests for the local browser UI glue."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from remove_ai_watermarks.webui import (
    ProcessOptions,
    UploadedFile,
    _archive_output_dir,
    _build_batch_cli_args,
    _LogCapture,
    _process_uploads,
    _safe_filename,
)


def test_safe_filename_removes_path_and_unsafe_chars() -> None:
    assert _safe_filename("../a weird 文件.png") == "a_weird_.png"
    assert _safe_filename("") == "image.png"


def test_build_batch_cli_args_uses_metadata_mode(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    options = ProcessOptions()

    assert _build_batch_cli_args(input_dir, output_dir, options) == [
        "batch",
        str(input_dir),
        "-o",
        str(output_dir),
        "--mode",
        "metadata",
    ]


def test_log_capture_forwards_chunks() -> None:
    chunks: list[str] = []
    capture = _LogCapture(chunks.append)

    capture.write("Loading model...\n")

    assert capture.getvalue() == "Loading model...\n"
    assert chunks == ["Loading model...\n"]


def test_process_uploads_reports_each_file(monkeypatch, tmp_path: Path) -> None:
    def fake_run_cli(args: list[str], **kwargs: object) -> tuple[int, str]:
        input_dir = Path(args[1])
        output_dir = Path(args[args.index("-o") + 1])
        output_dir.mkdir(exist_ok=True)
        for source in sorted(input_dir.iterdir()):
            (output_dir / source.name).write_bytes(b"clean")
        return 0, "done"

    monkeypatch.setattr("remove_ai_watermarks.webui._run_cli", fake_run_cli)
    options = ProcessOptions()

    batch = _process_uploads(
        tmp_path,
        [
            UploadedFile(filename="one.png", data=b"png"),
            UploadedFile(filename="two.txt", data=b"text"),
        ],
        options,
    )

    results = batch.results
    assert [result.status for result in results] == ["success", "error"]
    assert batch.archive_url is not None
    assert batch.archive_url.endswith("/processed_images.zip")
    assert results[0].output_url is not None
    assert results[0].output_name == "one_clean.png"
    assert "不支持" in results[1].message


def test_process_uploads_uses_batch_for_multiple_supported_files(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run_cli(args: list[str], **kwargs: object) -> tuple[int, str]:
        calls.append(args)
        assert args[0] == "batch"
        input_dir = Path(args[1])
        output_dir = Path(args[args.index("-o") + 1])
        output_dir.mkdir(exist_ok=True)
        for source in sorted(input_dir.iterdir()):
            (output_dir / source.name).write_bytes(b"clean")
        return 0, "batch done"

    monkeypatch.setattr("remove_ai_watermarks.webui._run_cli", fake_run_cli)
    options = ProcessOptions()

    batch = _process_uploads(
        tmp_path,
        [
            UploadedFile(filename="one.png", data=b"one"),
            UploadedFile(filename="two.jpg", data=b"two"),
        ],
        options,
    )

    assert len(calls) == 1
    assert calls[0][0] == "batch"
    assert [result.status for result in batch.results] == ["success", "success"]
    assert [result.output_name for result in batch.results] == ["one_clean.png", "two_clean.jpg"]
    assert batch.archive_url is not None


def test_archive_output_dir_contains_processed_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "one_clean.png").write_bytes(b"one")
    (output_dir / "two_clean.jpg").write_bytes(b"two")

    archive_bytes = _archive_output_dir(output_dir)

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == ["one_clean.png", "two_clean.jpg"]
        assert archive.read("one_clean.png") == b"one"
