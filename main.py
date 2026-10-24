"""Server web clipper: unggah video atau tempel link, dapat klip terbaik beserta subtitle."""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

import pipeline
import render

WORK_ROOT = Path(os.environ.get("WORK_DIR", "/tmp/clipper"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
JOB_TTL = int(os.environ.get("JOB_TTL_HOURS", "3")) * 3600
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1500"))

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
PIPE_SEM = threading.Semaphore(1)
RENDER_SEM = threading.Semaphore(1)

LANGS = {"id", "en", "auto"}
MODELS = {"tiny", "base", "small"}


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def _int(v, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _set(job: dict, **kw) -> None:
    with LOCK:
        job.update(kw)


def _cleanup() -> None:
    now = time.time()
    for jid in [k for k, j in JOBS.items() if now - j["created"] > JOB_TTL]:
        job = JOBS.pop(jid, None)
        if job:
            shutil.rmtree(job["dir"], ignore_errors=True)


def _public(job: dict) -> dict:
    keys = ("id", "status", "stage", "progress", "message", "error", "title", "duration", "language", "used_llm", "clips")
    out = {k: job.get(k) for k in keys}
    out["has_source"] = bool(job.get("source"))
    return out


def run_job(job: dict, url: str | None, params: dict) -> None:
    try:
        _set(job, status="queued", stage="queued", message="Menunggu antrean…", progress=0.0)
        with PIPE_SEM:
            wd = job["dir"]
            _set(job, status="running")
            if url:
                _set(job, stage="download", message="Mengunduh video…")
                path, title = pipeline.download(url, wd, lambda p: _set(job, progress=0.02 + 0.13 * p))
                _set(job, source=path, title=title)
            path = job["source"]
            duration = render.probe_duration(path)
            if duration > pipeline.MAX_MINUTES * 60:
                raise pipeline.PipelineError(f"Video {duration / 60:.0f} menit, melebihi batas {pipeline.MAX_MINUTES} menit.")
            _set(job, duration=round(duration, 1), stage="audio", message="Menyiapkan audio…", progress=0.16)
            wav = str(Path(wd) / "audio.wav")
            pipeline.extract_audio(path, wav)
            samples = pipeline.load_samples(wav)
            _set(job, stage="transcribe", message="Mendengarkan dan menulis transkrip…", progress=0.2)
            segments, words, lang = pipeline.transcribe(
                samples, params["language"], params["model"], duration, lambda p: _set(job, progress=0.2 + 0.7 * p)
            )
            if not segments:
                raise pipeline.PipelineError("Tidak ada ucapan yang terdeteksi di video ini.")
            _set(job, stage="analyze", message="Menilai potensi viral…", progress=0.92, language=lang)
            clips, used_llm = pipeline.analyze(
                segments, words, samples, params["clip_min"], params["clip_max"], params["count"], params["gemini_key"]
            )
            if not clips:
                raise pipeline.PipelineError("Video terlalu pendek atau ucapannya terlalu sedikit untuk dipotong.")
            _set(job, words=words, clips=clips, used_llm=used_llm, status="done", stage="done", message="Selesai", progress=1.0)
    except pipeline.PipelineError as e:
        _set(job, status="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set(job, status="error", error=f"Terjadi kesalahan di server: {type(e).__name__}: {str(e)[:300]}")


async def index(request: Request):
    return FileResponse(Path(__file__).with_name("index.html"), headers={"Cache-Control": "no-cache"})


async def health(request: Request):
    return JSONResponse({"ok": True})


async def create_job(request: Request):
    form = await request.form()
    url = (form.get("url") or "").strip() if isinstance(form.get("url"), str) else ""
    upload = form.get("file")
    has_file = upload is not None and not isinstance(upload, str) and getattr(upload, "filename", "")
    if not url and not has_file:
        return _err("Tempel link video atau pilih file dulu.")
    if url and not re.match(r"^https?://", url, re.I):
        return _err("Link harus diawali http:// atau https://")

    cmin = _int(form.get("clip_min"), 20, 10, 120)
    cmax = _int(form.get("clip_max"), 60, 15, 180)
    if cmax <= cmin:
        cmax = cmin + 15
    key = form.get("gemini_key")
    params = {
        "language": form.get("language") if form.get("language") in LANGS else "id",
        "model": form.get("model") if form.get("model") in MODELS else "base",
        "clip_min": cmin,
        "clip_max": cmax,
        "count": _int(form.get("count"), 8, 3, 15),
        "gemini_key": (key.strip() if isinstance(key, str) and key.strip() else os.environ.get("GEMINI_API_KEY")) or None,
    }

    _cleanup()
    jid = uuid.uuid4().hex[:12]
    wd = WORK_ROOT / jid
    wd.mkdir(parents=True, exist_ok=True)
    job = {
        "id": jid, "dir": str(wd), "created": time.time(), "status": "queued", "stage": "queued",
        "progress": 0.0, "message": "Menunggu antrean…", "error": None, "title": None, "duration": None,
        "language": None, "used_llm": False, "clips": [], "words": [], "source": None, "renders": {},
    }
    if has_file:
        ext = Path(upload.filename).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext or ""):
            ext = ".mp4"
        dest = wd / f"source{ext}"
        size = 0
        with open(dest, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    out.close()
                    shutil.rmtree(wd, ignore_errors=True)
                    return _err(f"File lebih dari {MAX_UPLOAD_MB} MB.", 413)
                out.write(chunk)
        job["source"] = str(dest)
        job["title"] = Path(upload.filename).stem[:80] or "Video"
        url = ""
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job, url or None, params), daemon=True).start()
    return JSONResponse({"id": jid})


def _get_job(request: Request):
    return JOBS.get(request.path_params["jid"])


async def job_status(request: Request):
    job = _get_job(request)
    if not job:
        return _err("Sesi tidak ditemukan atau sudah kedaluwarsa.", 404)
    with LOCK:
        return JSONResponse(_public(job))


async def job_source(request: Request):
    job = _get_job(request)
    if not job or not job.get("source") or not Path(job["source"]).exists():
        return _err("Video sumber tidak ada.", 404)
    return FileResponse(job["source"])


def run_render(job: dict, r: dict) -> None:
    try:
        with RENDER_SEM:
            _set(r, status="running")
            out = Path(job["dir"]) / f"klip_{r['id']}.mp4"
            render.render_clip(
                job["source"], str(out), r["start"], r["end"], job["words"],
                subtitles=r["subtitles"], style=r["style"], vertical=r["vertical"],
            )
            _set(r, status="done", file=str(out))
    except Exception as e:  # noqa: BLE001
        _set(r, status="error", error=str(e)[:300] or "Gagal membuat klip.")


async def create_render(request: Request):
    job = _get_job(request)
    if not job or job["status"] != "done":
        return _err("Analisis belum selesai.", 404)
    try:
        body = await request.json()
        start, end = float(body["start"]), float(body["end"])
    except Exception:  # noqa: BLE001
        return _err("Data klip tidak valid.")
    dur = job["duration"] or 0
    start = max(0.0, start)
    end = min(end, dur) if dur else end
    if end - start < 1 or end - start > 180:
        return _err("Panjang klip harus antara 1 dan 180 detik.")
    style = body.get("style") if body.get("style") in ("pop", "plain") else "pop"
    rid = uuid.uuid4().hex[:8]
    title = str(body.get("title") or "klip")
    r = {
        "id": rid, "status": "queued", "error": None, "file": None, "start": start, "end": end,
        "subtitles": bool(body.get("subtitles", True)), "style": style, "vertical": bool(body.get("vertical", True)),
        "name": re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()[:40] or "klip",
    }
    job["renders"][rid] = r
    threading.Thread(target=run_render, args=(job, r), daemon=True).start()
    return JSONResponse({"id": rid})


async def render_status(request: Request):
    job = _get_job(request)
    r = job and job["renders"].get(request.path_params["rid"])
    if not r:
        return _err("Klip tidak ditemukan.", 404)
    return JSONResponse({"status": r["status"], "error": r["error"]})


async def render_file(request: Request):
    job = _get_job(request)
    r = job and job["renders"].get(request.path_params["rid"])
    if not r or r["status"] != "done" or not Path(r["file"]).exists():
        return _err("Klip belum siap.", 404)
    return FileResponse(r["file"], media_type="video/mp4", filename=f"{r['name']}.mp4")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/health", health),
        Route("/api/jobs", create_job, methods=["POST"]),
        Route("/api/jobs/{jid}", job_status),
        Route("/api/jobs/{jid}/source", job_source),
        Route("/api/jobs/{jid}/render", create_render, methods=["POST"]),
        Route("/api/jobs/{jid}/render/{rid}", render_status),
        Route("/api/jobs/{jid}/render/{rid}/file", render_file),
    ]
)
