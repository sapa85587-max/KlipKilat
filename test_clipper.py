"""Uji cepat tanpa model: `python -m pytest -q` atau `python test_clipper.py`."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ["WORK_DIR"] = tempfile.mkdtemp(prefix="clipper-test-")

import pipeline  # noqa: E402
import render  # noqa: E402
import scoring  # noqa: E402

SENTENCES = [
    "Tahukah kamu bahwa 90 persen orang salah memilih waktu posting?",
    "Ini kesalahan yang bikin video kamu tenggelam, gila banget!",
    "Aku sudah coba 3 cara dan hanya satu yang berhasil.",
    "Pertama, pakai pembuka yang bikin penasaran dalam tiga detik.",
    "Kedua, potong semua jeda yang tidak perlu.",
    "Ketiga, tambahkan subtitle supaya orang bisa nonton tanpa suara.",
    "Hasilnya views naik 5 kali lipat dalam seminggu.",
    "Jadi, jangan pernah upload tanpa memikirkan pembuka videomu.",
    "Sekian dulu pembahasan hari ini dan terima kasih sudah menonton.",
    "Sampai jumpa lagi di video berikutnya ya teman teman semua.",
]


def fake_transcript():
    segments, words, t = [], [], 0.0
    for s in SENTENCES:
        toks = s.split()
        start = t
        for w in toks:
            words.append({"start": t, "end": t + 0.33, "word": " " + w})
            t += 0.36
        segments.append({"start": start, "end": t, "text": s})
        t += 0.4
    return segments, words


def make_video(path: str, seconds: int = 50) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", path],
        check=True,
    )


def test_scoring_prefers_strong_hook():
    segs = [
        {"start": 0, "end": 8, "text": "Tahukah kamu bahwa 90 persen orang salah memilih waktu posting?"},
        {"start": 8, "end": 18, "text": "Ini kesalahan yang bikin video kamu tenggelam, gila banget!"},
        {"start": 18, "end": 30, "text": "Aku sudah coba 3 cara dan hanya satu yang berhasil."},
        {"start": 30, "end": 40, "text": "lalu kemudian ya begitulah pokoknya seperti biasa saja"},
        {"start": 40, "end": 52, "text": "dan seterusnya sampai selesai begitu saja tanpa apa apa"},
    ]
    words = []
    for s in segs:
        toks = s["text"].split()
        step = (s["end"] - s["start"]) / len(toks)
        for i, w in enumerate(toks):
            words.append({"start": s["start"] + i * step, "end": s["start"] + (i + 1) * step - 0.02, "word": " " + w})
    cands = scoring.build_windows(segs, 20, 60)
    for c in cands:
        scoring.score_candidate(c, words)
    cands.sort(key=lambda c: c.score, reverse=True)
    assert cands[0].start == 0
    assert cands[0].score > cands[-1].score
    assert 0 <= cands[-1].score <= 100
    top = scoring.pick_top(cands, 3)
    for a in top:
        for b in top:
            if a is not b:
                assert min(a.end, b.end) - max(a.start, b.start) <= 0.3 * min(a.duration, b.duration)


def test_ass_generation():
    words = [{"start": 1.0 + i * 0.4, "end": 1.35 + i * 0.4, "word": " kata%d" % i} for i in range(7)]
    ass = render.make_ass(words, 1.0, 5.0, 1080, 1920, "pop")
    assert "PlayResX: 1080" in ass and ass.count("Dialogue:") == 3
    assert "\\k" in ass and "KATA0" in ass
    plain = render.make_ass(words, 1.0, 5.0, 1080, 1920, "plain")
    assert "\\k" not in plain and "kata0" in plain
    # kurung kurawal dan backslash dari transkrip tidak boleh menjadi tag ASS
    body = render.make_ass([{"start": 0, "end": 1, "word": "a{b}"}], 0, 2, 640, 360, "plain").split("[Events]")[1]
    assert "{" not in body and "a(b)" in body


def test_render_vertical_and_original():
    d = tempfile.mkdtemp()
    src = str(Path(d, "in.mp4"))
    make_video(src, 20)
    _, words = fake_transcript()
    v = str(Path(d, "v.mp4"))
    render.render_clip(src, v, 2.0, 12.0, words, True, "pop", True)
    assert render.probe_size(v) == (1080, 1920)
    assert abs(render.probe_duration(v) - 10.0) < 0.6
    o = str(Path(d, "o.mp4"))
    render.render_clip(src, o, 0.0, 6.0, words, True, "plain", False)
    w, h = render.probe_size(o)
    assert h == 360 and w == 640
    assert Path(d, "v.ass").exists()


def test_api_end_to_end(monkeypatch=None):
    from starlette.testclient import TestClient

    import main

    segments, words = fake_transcript()
    pipeline.transcribe = lambda wav, lang, model, dur, cb: (cb(1.0) or segments, words, "id")
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    d = tempfile.mkdtemp()
    src = str(Path(d, "in.mp4"))
    make_video(src, 50)

    assert client.post("/api/jobs", data={}).status_code == 400
    assert client.post("/api/jobs", data={"url": "ftp://x"}).status_code == 400
    with open(src, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("tes.mp4", f, "video/mp4")}, data={"clip_min": "15", "clip_max": "40", "count": "4"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(60):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    assert j["clips"] and all(0 <= c["score"] <= 100 for c in j["clips"])
    assert "words" not in j
    assert client.get(f"/api/jobs/{jid}/source").status_code == 200
    rng = client.get(f"/api/jobs/{jid}/source", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206

    c = j["clips"][0]
    r = client.post(f"/api/jobs/{jid}/render", json={"start": c["start"], "end": min(c["end"], c["start"] + 8), "subtitles": True, "style": "pop", "vertical": True, "title": c["title"]})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    for _ in range(120):
        s = client.get(f"/api/jobs/{jid}/render/{rid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert s["status"] == "done", s
    out = client.get(f"/api/jobs/{jid}/render/{rid}/file")
    assert out.status_code == 200 and out.headers["content-type"] == "video/mp4"
    assert "attachment" in out.headers["content-disposition"]
    assert len(out.content) > 10_000
    assert client.post(f"/api/jobs/{jid}/render", json={"start": 5, "end": 5.2}).status_code == 400
    assert client.get("/api/jobs/zzz").status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
