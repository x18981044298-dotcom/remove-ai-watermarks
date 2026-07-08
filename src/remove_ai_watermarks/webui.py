"""Small local browser UI for personal image processing."""

from __future__ import annotations

import html
import io
import json
import re
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlparse

import click

from remove_ai_watermarks.noai.constants import SUPPORTED_FORMATS

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_PORT = 8765
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    data: bytes


@dataclass(frozen=True)
class ProcessOptions:
    mode: str = "metadata"


@dataclass(frozen=True)
class ProcessResult:
    filename: str
    status: str
    exit_code: int
    message: str
    output_url: str | None
    output_name: str | None
    log: str


@dataclass(frozen=True)
class ProcessBatch:
    results: list[ProcessResult]
    archive_url: str | None


@dataclass
class JobState:
    id: str
    status: str
    message: str
    started_at: float
    log: str
    results: list[ProcessResult]
    archive_url: str | None
    error: str | None


class WebUiServer(ThreadingHTTPServer):
    """HTTP server that keeps the temp output directory alive."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(server_address, handler_class)
        self.workspace = Path(tempfile.mkdtemp(prefix="remove-ai-watermarks-ui-"))
        self.jobs: dict[str, JobState] = {}
        self.jobs_lock = threading.Lock()


def _safe_filename(filename: str) -> str:
    """Return a path-safe upload name, preserving the user's extension."""
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "image.png"
    sanitized = _SAFE_NAME_RE.sub("_", name).strip("._")
    return sanitized or "image.png"


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        next_candidate = directory / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _build_batch_cli_args(input_dir: Path, output_dir: Path, options: ProcessOptions) -> list[str]:
    return ["batch", str(input_dir), "-o", str(output_dir), "--mode", options.mode]


class _LogCapture(io.StringIO):
    """String buffer that forwards written chunks to a live progress callback."""

    def __init__(self, on_output: Callable[[str], None] | None) -> None:
        super().__init__()
        self._on_output = on_output

    def write(self, text: str) -> int:
        written = super().write(text)
        if self._on_output and text:
            self._on_output(text)
        return written


def _run_cli(args: list[str], on_output: Callable[[str], None] | None = None) -> tuple[int, str]:
    from remove_ai_watermarks.cli import main

    stdout = _LogCapture(on_output)
    stderr = _LogCapture(on_output)
    exit_code = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            main(args=args, prog_name="remove-ai-watermarks")
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                exit_code = code
            elif code is None:
                exit_code = 0
            else:
                exit_code = 1
    log = stdout.getvalue()
    err = stderr.getvalue()
    if err:
        log = f"{log}\n{err}" if log else err
    return exit_code, log.strip()


def _classify_result(exit_code: int, output_path: Path) -> tuple[str, str]:
    output_exists = output_path.exists()
    if exit_code == 0 and output_exists:
        return "success", "处理完成"
    if exit_code == 0:
        return "skipped", "没有生成输出文件"
    if exit_code == 2:
        return "skipped", "已跳过, 未检测到对应水印信号"
    if output_exists:
        return "warning", "部分完成, 请查看提示"
    return "error", "处理失败"


def _process_uploads_batch(
    workspace: Path,
    uploads: list[UploadedFile],
    options: ProcessOptions,
    on_output: Callable[[str], None] | None = None,
) -> ProcessBatch:
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=workspace))
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    indexed_results: list[tuple[int, ProcessResult]] = []
    supported: list[tuple[int, UploadedFile, Path]] = []
    for index, upload in enumerate(uploads):
        safe_name = _safe_filename(upload.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            indexed_results.append(
                (
                    index,
                    ProcessResult(
                        filename=upload.filename,
                        status="error",
                        exit_code=1,
                        message=f"不支持的文件格式: {suffix or '无扩展名'}",
                        output_url=None,
                        output_name=None,
                        log="",
                    ),
                )
            )
            continue

        source = _unique_path(input_dir, safe_name)
        source.write_bytes(upload.data)
        supported.append((index, upload, source))

    if supported:
        exit_code, log = _run_cli(_build_batch_cli_args(input_dir, output_dir, options), on_output=on_output)
        for index, upload, source in supported:
            batch_output = output_dir / source.name
            final_output = batch_output
            if batch_output.exists():
                final_output = _unique_path(output_dir, f"{source.stem}_clean{source.suffix}")
                batch_output.rename(final_output)
            status, message = _classify_result(exit_code, final_output)
            rel_url = None
            final_name = None
            if final_output.exists():
                rel_url = f"/outputs/{run_dir.name}/{final_output.name}"
                final_name = final_output.name
            indexed_results.append(
                (
                    index,
                    ProcessResult(
                        filename=upload.filename,
                        status=status,
                        exit_code=exit_code,
                        message=message,
                        output_url=rel_url,
                        output_name=final_name,
                        log=log,
                    ),
                )
            )

    results = [result for _index, result in sorted(indexed_results, key=lambda item: item[0])]
    has_outputs = any(result.output_url is not None for result in results)
    archive_url = f"/archives/{run_dir.name}/processed_images.zip" if has_outputs else None
    return ProcessBatch(results=results, archive_url=archive_url)


def _process_uploads(
    workspace: Path,
    uploads: list[UploadedFile],
    options: ProcessOptions,
    on_output: Callable[[str], None] | None = None,
) -> ProcessBatch:
    return _process_uploads_batch(workspace, uploads, options, on_output=on_output)


def _archive_output_dir(output_dir: Path) -> bytes:
    """Zip all processed files in *output_dir* and return the archive bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_file in sorted(p for p in output_dir.iterdir() if p.is_file()):
            archive.write(output_file, arcname=output_file.name)
    return buffer.getvalue()


def _parse_multipart(content_type: str, body: bytes) -> tuple[list[UploadedFile], dict[str, str]]:
    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8", errors="replace") + body
    )
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    if not message.is_multipart():
        return [], {}

    files: list[UploadedFile] = []
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str):
            continue
        raw_payload = part.get_payload(decode=True)
        payload = raw_payload if isinstance(raw_payload, bytes) else b""
        filename = part.get_filename()
        if filename:
            files.append(UploadedFile(filename=filename, data=payload))
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return files, fields


def _options_from_fields(fields: dict[str, str]) -> ProcessOptions:
    return ProcessOptions()


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _html_response(handler: BaseHTTPRequestHandler, body: str) -> None:
    data = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _job_payload(job: JobState) -> dict[str, Any]:
    elapsed = max(0, int(time.monotonic() - job.started_at))
    return {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "elapsed": elapsed,
        "log": job.log,
        "archive_url": job.archive_url,
        "results": [result.__dict__ for result in job.results],
        "error": job.error,
    }


def _append_job_log(server: WebUiServer, job_id: str, text: str) -> None:
    if not text:
        return
    with server.jobs_lock:
        job = server.jobs[job_id]
        job.log += text
        if len(job.log) > 80_000:
            job.log = job.log[-80_000:]


def _start_job(server: WebUiServer, files: list[UploadedFile], options: ProcessOptions) -> JobState:
    job_id = uuid.uuid4().hex
    job = JobState(
        id=job_id,
        status="running",
        message="任务已开始, 正在准备图片...",
        started_at=time.monotonic(),
        log="任务已开始。\n",
        results=[],
        archive_url=None,
        error=None,
    )
    with server.jobs_lock:
        server.jobs[job_id] = job

    def worker() -> None:
        try:
            with server.jobs_lock:
                job.message = "正在移除 AI 元数据..."
            batch = _process_uploads(
                server.workspace,
                files,
                options,
                on_output=lambda text: _append_job_log(server, job_id, text),
            )
            with server.jobs_lock:
                job.status = "done"
                job.message = "处理完成"
                job.results = batch.results
                job.archive_url = batch.archive_url
        except Exception as exc:  # pragma: no cover - defensive job boundary
            with server.jobs_lock:
                job.status = "error"
                job.message = "处理失败"
                job.error = str(exc)
                job.log += f"\nError: {exc}\n"

    thread = threading.Thread(target=worker, name=f"remove-ai-watermarks-ui-{job_id[:8]}", daemon=True)
    thread.start()
    return job


class WebUiHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            _html_response(self, _INDEX_HTML)
            return
        if parsed.path.startswith("/outputs/"):
            self._serve_output(parsed.path)
            return
        if parsed.path.startswith("/archives/"):
            self._serve_archive(parsed.path)
            return
        if parsed.path.startswith("/jobs/"):
            self._serve_job(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_UPLOAD_BYTES:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "上传内容为空或超过 512 MB"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求格式不是 multipart/form-data"})
            return

        files, fields = _parse_multipart(content_type, self.rfile.read(length))
        if not files:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请选择至少一张图片"})
            return

        job = _start_job(self._server, files, _options_from_fields(fields))
        _json_response(self, HTTPStatus.ACCEPTED, {"job_id": job.id})

    def _serve_job(self, request_path: str) -> None:
        parts = [unquote(part) for part in request_path.split("/") if part]
        if len(parts) != 2:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, job_id = parts
        with self._server.jobs_lock:
            job = self._server.jobs.get(job_id)
            payload = _job_payload(job) if job else None
        if payload is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _json_response(self, HTTPStatus.OK, payload)

    def _serve_output(self, request_path: str) -> None:
        parts = [unquote(part) for part in request_path.split("/") if part]
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, run_name, filename = parts
        safe_run = _safe_filename(run_name)
        safe_file = _safe_filename(filename)
        output_path = self._server.workspace / safe_run / "output" / safe_file
        try:
            output_path.relative_to(self._server.workspace)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not output_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        data = output_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{html.escape(output_path.name)}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_archive(self, request_path: str) -> None:
        parts = [unquote(part) for part in request_path.split("/") if part]
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, run_name, filename = parts
        if _safe_filename(filename) != "processed_images.zip":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        safe_run = _safe_filename(run_name)
        output_dir = self._server.workspace / safe_run / "output"
        try:
            output_dir.relative_to(self._server.workspace)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not output_dir.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        data = _archive_output_dir(output_dir)
        if not data:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="processed_images.zip"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @property
    def _server(self) -> WebUiServer:
        return cast("WebUiServer", self.server)


def serve(host: str = "127.0.0.1", port: int = _DEFAULT_PORT, *, open_browser: bool = True) -> None:
    """Start the local browser UI and block until interrupted."""
    server = WebUiServer((host, port), WebUiHandler)
    actual_host, actual_port = cast("tuple[str, int]", server.server_address)
    url = f"http://{actual_host}:{actual_port}/"
    click.echo(f"Remove-AI-Watermarks UI: {url}")
    click.echo("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()


def main() -> None:
    serve()


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 元数据清理</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1d2733;
      --muted: #687385;
      --line: #d9dee7;
      --accent: #246bfe;
      --accent-dark: #164dc1;
      --ok: #0f7b4f;
      --warn: #a35d00;
      --bad: #b42318;
      --skip: #4e5d70;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 32px auto;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }
    p { margin: 0; color: var(--muted); }
    .panel {
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }
    .notice {
      margin-top: 16px;
      padding: 12px 14px;
      border: 1px solid #f0d498;
      border-radius: 6px;
      background: #fff8e8;
      color: #634100;
      font-size: 14px;
    }
    label {
      display: block;
      margin: 0 0 8px;
      font-weight: 650;
    }
    input[type="file"] {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
    }
    .drop-zone {
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px dashed #aab4c3;
      border-radius: 6px;
      background: #fbfcfe;
      text-align: center;
      cursor: pointer;
      padding: 22px;
    }
    .drop-zone:focus,
    .drop-zone.dragging {
      outline: 2px solid rgba(36, 107, 254, 0.25);
      border-color: var(--accent);
      background: #f4f8ff;
    }
    .drop-title {
      font-weight: 700;
      color: var(--ink);
    }
    .drop-hint {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    .queue {
      display: grid;
      gap: 8px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
    }
    .queue-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fbfcfe;
    }
    .queue-name {
      min-width: 0;
      overflow-wrap: anywhere;
      color: var(--ink);
    }
    button {
      margin-top: 18px;
      min-height: 44px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { cursor: wait; opacity: 0.68; }
    .results {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .progress {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 14px;
    }
    .progress-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .progress-title {
      font-weight: 700;
    }
    .progress-time {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .progress-bar {
      position: relative;
      height: 8px;
      margin-top: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #e8edf5;
    }
    .progress-bar::before {
      content: "";
      position: absolute;
      inset: 0;
      width: 36%;
      border-radius: inherit;
      background: var(--accent);
      animation: slide 1.1s ease-in-out infinite;
    }
    @keyframes slide {
      0% { transform: translateX(-110%); }
      50% { transform: translateX(120%); }
      100% { transform: translateX(280%); }
    }
    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 14px;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .name {
      min-width: 0;
      overflow-wrap: anywhere;
      font-weight: 700;
    }
    .badge {
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      color: white;
    }
    .success { background: var(--ok); }
    .warning { background: var(--warn); }
    .error { background: var(--bad); }
    .skipped { background: var(--skip); }
    .message {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .actions {
      margin-top: 10px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .toolbar {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
    }
    .download-all {
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
    }
    .download-all:hover {
      background: var(--accent-dark);
      text-decoration: none;
    }
    a {
      color: var(--accent-dark);
      font-weight: 650;
      text-decoration: none;
    }
    a:hover { text-decoration: underline; }
    details {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 260px;
      overflow: auto;
      padding: 10px;
      border-radius: 6px;
      background: #f3f5f8;
      color: #2f3a4a;
    }
    @media (max-width: 680px) {
      main { width: min(100vw - 20px, 920px); margin: 18px auto; }
      .grid { grid-template-columns: 1fr; }
      .panel { padding: 16px; }
      h1 { font-size: 23px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>AI 元数据清理</h1>
    <p>本地浏览器界面, 支持选择文件、拖拽图片, 也支持直接复制后 Ctrl+V 粘贴。</p>

    <div class="notice">
      当前版本只移除 AI 元数据, 不处理可见水印, 不运行 SDXL/ControlNet, 也不会重新生成图片像素。
    </div>

    <form id="form" class="panel">
      <label for="fileInput">添加图片</label>
      <input id="fileInput" name="files" type="file" accept="image/*" multiple>
      <div id="dropZone" class="drop-zone" tabindex="0" role="button" aria-label="添加图片">
        <div>
          <div class="drop-title">点击选择图片, 或把图片拖到这里</div>
          <div class="drop-hint">也可以先复制图片, 然后在这个页面按 Ctrl+V 粘贴</div>
        </div>
      </div>
      <div id="queue" class="queue"></div>

      <button id="submit" type="submit">开始处理</button>
    </form>

    <section id="results" class="results" aria-live="polite"></section>
  </main>

  <script>
    const form = document.getElementById("form");
    const submit = document.getElementById("submit");
    const results = document.getElementById("results");
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const queue = document.getElementById("queue");
    const selectedFiles = [];

    const labels = {
      success: "完成",
      warning: "部分完成",
      skipped: "跳过",
      error: "失败"
    };

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function fileKey(file) {
      return `${file.name}|${file.size}|${file.lastModified}`;
    }

    function addFiles(files) {
      const existing = new Set(selectedFiles.map(fileKey));
      for (const file of files) {
        if (!file.type.startsWith("image/")) {
          continue;
        }
        const normalized = file.name
          ? file
          : new File(
              [file],
              `pasted-${Date.now()}-${selectedFiles.length + 1}.png`,
              { type: file.type || "image/png" }
            );
        const key = fileKey(normalized);
        if (!existing.has(key)) {
          selectedFiles.push(normalized);
          existing.add(key);
        }
      }
      renderQueue();
    }

    function renderQueue() {
      if (!selectedFiles.length) {
        queue.innerHTML = `<div class="message">还没有添加图片。</div>`;
        return;
      }
      queue.innerHTML = selectedFiles.map((file, index) => `
        <div class="queue-item">
          <span class="queue-name">${escapeHtml(file.name)}</span>
          <span>${Math.max(1, Math.round(file.size / 1024))} KB</span>
        </div>
      `).join("");
    }

    function renderResult(item) {
      const link = item.output_url
        ? `<a href="${escapeHtml(item.output_url)}" download>下载处理后图片</a>`
        : "";
      const log = item.log
        ? `<details><summary>查看日志</summary><pre>${escapeHtml(item.log)}</pre></details>`
        : "";
      return `
        <article class="result">
          <div class="result-head">
            <div class="name">${escapeHtml(item.filename)}</div>
            <span class="badge ${escapeHtml(item.status)}">${labels[item.status] || item.status}</span>
          </div>
          <div class="message">${escapeHtml(item.message)}, 退出码 ${escapeHtml(item.exit_code)}</div>
          <div class="actions">${link}</div>
          ${log}
        </article>
      `;
    }

    function renderArchiveLink(archiveUrl) {
      if (!archiveUrl) {
        return "";
      }
      return `
        <div class="toolbar">
          <a class="download-all" href="${escapeHtml(archiveUrl)}" download>下载全部 ZIP</a>
        </div>
      `;
    }

    function renderProgress(job) {
      const log = job.log
        ? `<details open><summary>实时日志</summary><pre>${escapeHtml(job.log)}</pre></details>`
        : "";
      return `
        <article class="progress">
          <div class="progress-head">
            <div>
              <div class="progress-title">${escapeHtml(job.message || "正在处理...")}</div>
              <div class="message">只清理 C2PA、EXIF、PNG 文本块等 AI 元数据, 不重绘图片像素。</div>
            </div>
            <div class="progress-time">${escapeHtml(job.elapsed || 0)} 秒</div>
          </div>
          <div class="progress-bar" aria-hidden="true"></div>
          ${log}
        </article>
      `;
    }

    async function pollJob(jobId) {
      while (true) {
        const response = await fetch(`/jobs/${encodeURIComponent(jobId)}`);
        const job = await response.json();
        if (!response.ok) {
          throw new Error(job.error || "任务查询失败");
        }
        if (job.status === "running") {
          results.innerHTML = renderProgress(job);
          await new Promise((resolve) => setTimeout(resolve, 1000));
          continue;
        }
        if (job.status === "done") {
          results.innerHTML = renderArchiveLink(job.archive_url) + job.results.map(renderResult).join("");
          return;
        }
        results.innerHTML = `
          <article class="result">
            <div class="result-head">
              <div class="name">处理失败</div>
              <span class="badge error">失败</span>
            </div>
            <div class="message">${escapeHtml(job.error || job.message || "未知错误")}</div>
            ${job.log ? `<details open><summary>实时日志</summary><pre>${escapeHtml(job.log)}</pre></details>` : ""}
          </article>
        `;
        return;
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!selectedFiles.length) {
        results.innerHTML = `
          <article class="result">
            <div class="result-head">
              <div class="name">请先添加图片</div>
              <span class="badge skipped">等待</span>
            </div>
          </article>
        `;
        return;
      }
      const data = new FormData();
      selectedFiles.forEach((file) => data.append("files", file, file.name));
      submit.disabled = true;
      submit.textContent = "处理中...";
      results.innerHTML = `<article class="progress"><div class="progress-title">任务提交中...</div></article>`;
      try {
        const response = await fetch("/process", { method: "POST", body: data });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "请求失败");
        }
        await pollJob(payload.job_id);
      } catch (error) {
        results.innerHTML = `
          <article class="result">
            <div class="result-head">
              <div class="name">请求失败</div>
              <span class="badge error">失败</span>
            </div>
            <div class="message">${escapeHtml(error.message || error)}</div>
          </article>
        `;
      } finally {
        submit.disabled = false;
        submit.textContent = "开始处理";
      }
    });

    fileInput.addEventListener("change", () => {
      addFiles(Array.from(fileInput.files || []));
      fileInput.value = "";
    });

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropZone.classList.add("dragging");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragging");
    });
    dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
      addFiles(Array.from(event.dataTransfer.files || []));
    });

    document.addEventListener("paste", (event) => {
      const items = Array.from(event.clipboardData?.items || []);
      const pasted = items
        .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
        .map((item) => item.getAsFile())
        .filter(Boolean);
      if (pasted.length) {
        event.preventDefault();
        addFiles(pasted);
      }
    });

    renderQueue();
  </script>
</body>
</html>
"""
