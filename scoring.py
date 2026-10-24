"""Cari kandidat klip dari transkrip lalu nilai potensi viralnya.

Penilaian ini perkiraan berbasis ciri teks dan suara (pembuka, angka, emosi,
tempo bicara, jeda, energi suara). Skor tidak menjamin sebuah klip akan viral.
Jika ada kunci Gemini, 12 kandidat teratas dinilai ulang oleh model bahasa.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

HOOK_PHRASES = [
    # Indonesia
    "tahukah kamu", "tahu nggak", "tahu gak", "pernah nggak", "pernah gak", "rahasia", "jangan pernah",
    "kesalahan", "ternyata", "faktanya", "kenyataannya", "bayangkan", "kenapa", "mengapa",
    "gimana caranya", "bagaimana cara", "cara ", "stop ", "hati-hati", "awas", "yang jarang",
    "percaya atau tidak", "serius", "ini dia", "inilah", "alasan ", "buktinya", "yang paling",
    # Inggris
    "did you know", "the truth", "secret", "nobody", "no one", "never ", "mistake", "here's why",
    "what if", "imagine", "stop ", "the biggest", "the real reason", "you won't believe", "how to",
]

EMOTION_WORDS = [
    "gila", "luar biasa", "parah", "bahaya", "takut", "marah", "sedih", "kaget", "wow", "hebat",
    "mengejutkan", "menakutkan", "menyakitkan", "bangga", "malu", "berhasil", "gagal", "bangkrut",
    "kaya", "miskin", "cinta", "benci", "sakit", "menang", "kalah", "mustahil", "banget", "sumpah",
    "amazing", "shocking", "incredible", "insane", "crazy", "terrible", "love", "hate", "fear",
    "angry", "proud", "failed", "broke", "rich", "impossible", "huge", "worst", "best",
]

NUMBER_RE = re.compile(r"\d|\b(?:juta|miliar|triliun|ribu|persen|percent|million|billion|thousand)\b|%|rp\b|\$", re.I)
SENTENCE_END = re.compile(r"[.!?…]['\")\]]*\s*$")

WEIGHTS = {
    "hook": 0.26,
    "emosi": 0.14,
    "spesifik": 0.10,
    "tempo": 0.14,
    "jeda": 0.10,
    "energi": 0.10,
    "utuh": 0.10,
    "durasi": 0.06,
}


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    features: dict = field(default_factory=dict)
    score: float = 0.0
    title: str = ""
    reasons: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _count_hits(text: str, vocab: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in vocab if w in t)


def build_windows(segments: list[dict], min_len: float, max_len: float) -> list[Candidate]:
    """Satu kandidat per awal kalimat: tambahkan segmen sampai cukup panjang dan berakhir di akhir kalimat."""
    out: list[Candidate] = []
    n = len(segments)
    for i in range(n):
        start = segments[i]["start"]
        text_parts: list[str] = []
        end = start
        for j in range(i, n):
            seg = segments[j]
            if seg["end"] - start > max_len and j > i:
                break
            text_parts.append(seg["text"].strip())
            end = seg["end"]
            dur = end - start
            if dur >= min_len and SENTENCE_END.search(seg["text"]):
                break
        dur = end - start
        if dur < min_len * 0.7:
            continue
        out.append(Candidate(start=start, end=end, text=" ".join(text_parts).strip()))
    return out


def _words_in(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]


def energy_profile(samples, sample_rate: int = 16000, block_s: float = 0.5):
    """RMS per blok 0.5 detik. `samples` adalah array numpy float32."""
    import numpy as np

    block = int(sample_rate * block_s)
    if len(samples) < block:
        return np.zeros(1), 1.0
    n = len(samples) // block
    blocks = samples[: n * block].reshape(n, block)
    rms = np.sqrt((blocks.astype("float64") ** 2).mean(axis=1))
    global_mean = float(rms.mean()) or 1.0
    return rms, global_mean


def score_candidate(c: Candidate, words: list[dict], energy=None, block_s: float = 0.5) -> Candidate:
    ws = _words_in(words, c.start, c.end)
    text = c.text
    first = " ".join(w["word"].strip() for w in ws[:14]) or text[:120]
    f: dict[str, float] = {}
    reasons: list[str] = []

    # Pembuka
    hook = 0.0
    if "?" in first:
        hook += 0.45
        reasons.append("Dibuka dengan pertanyaan")
    hits = _count_hits(first, HOOK_PHRASES)
    if hits:
        hook += min(0.55, 0.3 * hits)
        reasons.append("Kalimat pembuka memancing rasa penasaran")
    if NUMBER_RE.search(first):
        hook += 0.15
    f["hook"] = _clamp(hook)

    # Emosi
    emo_hits = _count_hits(text, EMOTION_WORDS)
    excl = text.count("!")
    per_min = (emo_hits + excl * 0.5) / max(c.duration / 60.0, 0.3)
    f["emosi"] = _clamp(per_min / 6.0)
    if f["emosi"] > 0.5:
        reasons.append("Bahasa emosional dan bertenaga")

    # Spesifik
    nums = len(NUMBER_RE.findall(text))
    f["spesifik"] = _clamp(nums / 4.0)
    if nums >= 3:
        reasons.append("Banyak angka dan fakta spesifik")

    # Tempo bicara (kata per detik)
    if ws:
        spoken = max(ws[-1]["end"] - ws[0]["start"], 1.0)
        wps = len(ws) / spoken
    else:
        wps = len(text.split()) / max(c.duration, 1.0)
    f["tempo"] = _clamp(1.0 - abs(wps - 2.9) / 1.6)
    if f["tempo"] > 0.75:
        reasons.append("Tempo bicara pas dan enak diikuti")

    # Jeda panjang
    pause = 0.0
    for a, b in zip(ws, ws[1:]):
        gap = b["start"] - a["end"]
        if gap > 0.8:
            pause += gap
    ratio = pause / max(c.duration, 1.0)
    f["jeda"] = _clamp(1.0 - ratio * 5.0)
    if f["jeda"] < 0.5:
        reasons.append("Banyak jeda panjang (menurunkan skor)")

    # Energi suara
    if energy is not None:
        rms, gmean = energy
        a = int(c.start / block_s)
        b = max(int(c.end / block_s), a + 1)
        seg = rms[a:b]
        if len(seg):
            rel = float(seg.mean()) / gmean
            var = float(seg.std()) / (float(seg.mean()) or 1.0)
            f["energi"] = _clamp(0.55 * _clamp((rel - 0.7) / 0.8) + 0.45 * _clamp(var / 0.8))
            if f["energi"] > 0.65:
                reasons.append("Suara dinamis dan penuh energi")
        else:
            f["energi"] = 0.5
    else:
        f["energi"] = 0.5

    # Utuh: mulai dan berakhir di batas kalimat
    starts_ok = bool(text[:1].isupper() or text[:1].isdigit())
    ends_ok = bool(SENTENCE_END.search(text))
    f["utuh"] = 0.5 * starts_ok + 0.5 * ends_ok
    if f["utuh"] == 1.0:
        reasons.append("Mulai dan berakhir di batas kalimat")

    # Durasi ideal 25-55 detik
    d = c.duration
    if 25 <= d <= 55:
        f["durasi"] = 1.0
    elif d < 25:
        f["durasi"] = _clamp(d / 25.0)
    else:
        f["durasi"] = _clamp(1.0 - (d - 55) / 40.0)

    c.features = {k: round(v, 3) for k, v in f.items()}
    c.score = round(100 * sum(WEIGHTS[k] * f[k] for k in WEIGHTS), 1)
    c.reasons = reasons[:4]
    c.title = make_title(text)
    return c


def make_title(text: str, limit: int = 64) -> str:
    sentence = re.split(r"(?<=[.!?…])\s+", text.strip(), maxsplit=1)[0]
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:-") + "…"


def pick_top(cands: list[Candidate], count: int, overlap_limit: float = 0.3) -> list[Candidate]:
    chosen: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        ok = True
        for k in chosen:
            inter = min(c.end, k.end) - max(c.start, k.start)
            if inter > 0 and inter / min(c.duration, k.duration) > overlap_limit:
                ok = False
                break
        if ok:
            chosen.append(c)
        if len(chosen) >= count:
            break
    return chosen


def label_for(score: float) -> str:
    if score >= 72:
        return "Potensi tinggi"
    if score >= 55:
        return "Potensi sedang"
    return "Potensi rendah"


def llm_rerank(cands: list[Candidate], api_key: str, model: str = "gemini-2.5-flash", timeout: float = 40.0) -> bool:
    """Nilai ulang kandidat lewat Gemini. Mengembalikan True jika berhasil. Gagal diam-diam ke skor heuristik."""
    if not api_key or not cands:
        return False
    items = [
        {"id": i, "durasi_detik": round(c.duration), "teks": c.text[:900]}
        for i, c in enumerate(cands)
    ]
    prompt = (
        "Kamu editor video pendek (Reels, TikTok, YouTube Shorts). Nilai tiap potongan transkrip di bawah "
        "dari 0 sampai 100 untuk potensi ditonton sampai habis dan dibagikan. Pertimbangkan kekuatan 3 detik "
        "pertama, rasa penasaran, emosi, nilai informasi, dan apakah potongan bisa dipahami berdiri sendiri. "
        "Balas hanya JSON berupa array objek {\"id\": number, \"score\": number, \"title\": string, "
        "\"reason\": string}. Title maksimal 60 karakter dalam bahasa yang sama dengan transkrip. "
        "Reason satu kalimat singkat dalam bahasa Indonesia.\n\n" + json.dumps(items, ensure_ascii=False)
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
        }
    ).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        rated = json.loads(raw)
    except Exception:
        return False
    ok = False
    for r in rated if isinstance(rated, list) else []:
        try:
            c = cands[int(r["id"])]
            llm_score = _clamp(float(r["score"]), 0, 100)
        except Exception:
            continue
        c.score = round(0.5 * c.score + 0.5 * llm_score, 1)
        if r.get("title"):
            c.title = str(r["title"])[:80]
        if r.get("reason"):
            c.reasons = [str(r["reason"])[:140]] + c.reasons[:3]
        ok = True
    return ok
