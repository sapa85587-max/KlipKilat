"""Potong klip dengan ffmpeg dan tempel subtitle (ASS) langsung ke video."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _ts(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str) -> str:
    return word.replace("{", "(").replace("}", ")").replace("\\", "").replace("\n", " ").strip()


def group_words(words: list[dict], max_words: int = 3, max_gap: float = 0.7) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            ends = cur[-1]["word"].strip().endswith((".", "?", "!", ","))
            if len(cur) >= max_words or gap > max_gap or ends:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def make_ass(words: list[dict], clip_start: float, clip_end: float, width: int, height: int, style: str = "pop") -> str:
    """style: 'pop' (kata aktif menyala kuning) atau 'plain' (putih polos)."""
    ws = [w for w in words if w["end"] > clip_start and w["start"] < clip_end and _clean(w["word"])]
    size = int(height * (0.046 if height >= width else 0.07))
    margin_v = int(height * (0.20 if height >= width else 0.10))
    outline = max(3, int(size * 0.09))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        # Warna ASS berformat &HAABBGGRR. Primary = warna setelah disorot, Secondary = sebelum disorot.
        f"Style: Default,DejaVu Sans,{size},&H0000E6FF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,"
        f"{outline},2,2,{int(width * 0.06)},{int(width * 0.06)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    groups = group_words(ws)
    for gi, g in enumerate(groups):
        s = max(g[0]["start"], clip_start) - clip_start
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else g[-1]["end"]
        e = min(max(g[-1]["end"], min(nxt, g[-1]["end"] + 0.4)), clip_end) - clip_start
        if e <= s:
            continue
        if style == "pop":
            parts = []
            for wi, w in enumerate(g):
                w_end = g[wi + 1]["start"] if wi + 1 < len(g) else w["end"]
                cs = max(1, int(round((w_end - max(w["start"], clip_start)) * 100)))
                parts.append(f"{{\\k{cs}}}{_clean(w['word']).upper()}")
            text = " ".join(parts)
        else:
            text = " ".join(_clean(w["word"]) for w in g)
        lines.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Default,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def render_clip(
    src: str,
    out_path: str,
    start: float,
    end: float,
    words: list[dict],
    subtitles: bool = True,
    style: str = "pop",
    vertical: bool = True,
    workdir: str | None = None,
) -> None:
    workdir = workdir or str(Path(out_path).parent)
    dur = max(end - start, 0.5)
    if vertical:
        # Potong tengah ke 9:16 lalu skala ke 1080x1920. Aman untuk video yang sudah vertikal.
        out_w, out_h = 1080, 1920
        vf = [f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'", f"scale={out_w}:{out_h}"]
    else:
        sw, sh = probe_size(src)
        out_h = min(sh, 720)
        out_h -= out_h % 2
        out_w = int(round(sw * out_h / sh / 2) * 2)
        vf = [f"scale={out_w}:{out_h}"]

    if subtitles:
        ass_name = Path(out_path).stem + ".ass"
        Path(workdir, ass_name).write_text(make_ass(words, start, end, out_w, out_h, style), encoding="utf-8")
        vf.append(f"ass={ass_name}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-threads", "2",
        Path(out_path).name,
    ]
    res = subprocess.run(cmd, cwd=Path(out_path).parent, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[-600:] or "ffmpeg gagal")
