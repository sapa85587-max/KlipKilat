"""Unduh video, ekstrak audio, transkripsi, lalu cari dan nilai klip."""
from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path
from typing import Callable

import scoring

MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "90"))
_MODEL_CACHE: dict = {}


class PipelineError(Exception):
    pass


def download(url: str, workdir: str, on_progress: Callable[[float], None]) -> tuple[str, str]:
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise PipelineError("yt-dlp belum terpasang di server.") from e

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    def too_long(info, *, incomplete):
        dur = info.get("duration")
        if dur and dur > MAX_MINUTES * 60:
            return f"Video {dur / 60:.0f} menit, melebihi batas {MAX_MINUTES} menit."
        return None

    opts = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(Path(workdir) / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "match_filter": too_long,
        "retries": 3,
    }
    cookies = os.environ.get("YT_COOKIES")
    if cookies:
        cpath = Path(workdir) / "cookies.txt"
        cpath.write_text(cookies)
        opts["cookiefile"] = str(cpath)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        msg = str(e)
        if "Sign in" in msg or "bot" in msg.lower():
            raise PipelineError(
                "Situs menolak unduhan dari server ini. Unduh videonya ke HP lalu unggah sebagai file."
            ) from e
        raise PipelineError("Gagal mengunduh dari link itu. " + msg.splitlines()[-1][:200]) from e
    files = sorted(Path(workdir).glob("source.*"))
    files = [f for f in files if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise PipelineError("Video tidak terunduh. Coba unggah file langsung.")
    return str(files[0]), (info or {}).get("title") or "Video"


def extract_audio(video: str, wav: str) -> None:
    res = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", wav],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise PipelineError("File itu tidak punya audio yang bisa dibaca.")


def load_samples(wav: str):
    import numpy as np

    with wave.open(wav, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0


def transcribe(audio, language: str, model_size: str, duration: float, on_progress: Callable[[float], None]):
    """`audio` adalah array numpy float32 mono 16 kHz, supaya faster-whisper tidak memakai PyAV untuk membaca file."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise PipelineError("faster-whisper belum terpasang di server.") from e
    model = _MODEL_CACHE.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _MODEL_CACHE.clear()
        _MODEL_CACHE[model_size] = model
    lang = None if language == "auto" else language
    seg_iter, info = model.transcribe(
        audio, language=lang, word_timestamps=True, vad_filter=True, beam_size=1, condition_on_previous_text=False
    )
    segments, words = [], []
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        segments.append({"start": float(s.start), "end": float(s.end), "text": text})
        for w in s.words or []:
            words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
        if duration:
            on_progress(min(s.end / duration, 1.0))
    return segments, words, info.language


def analyze(segments, words, samples, clip_min: float, clip_max: float, count: int, gemini_key: str | None):
    energy = scoring.energy_profile(samples) if samples is not None else None
    cands = scoring.build_windows(segments, clip_min, clip_max)
    for c in cands:
        scoring.score_candidate(c, words, energy)
    top = scoring.pick_top(cands, count)
    used_llm = scoring.llm_rerank(top, gemini_key) if gemini_key else False
    top.sort(key=lambda c: c.score, reverse=True)
    clips = []
    for i, c in enumerate(top):
        clips.append(
            {
                "id": i,
                "start": round(c.start, 2),
                "end": round(c.end, 2),
                "duration": round(c.duration, 1),
                "score": c.score,
                "label": scoring.label_for(c.score),
                "title": c.title,
                "reasons": c.reasons,
                "text": c.text[:260],
                "features": c.features,
            }
        )
    return clips, used_llm
