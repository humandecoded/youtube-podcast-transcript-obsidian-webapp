# services/podcast_notes.py
"""
Podcast → Obsidian utilities

- Input: podcast episode URL (Apple/Spotify/site/host page or RSS <item> link)
- Metadata: via yt-dlp (no media download)
- Transcript:
    1) Scrape episode page for transcript blocks/links
    2) Discover RSS <link rel="alternate" type="application/rss+xml">; look for Podcasting 2.0 <podcast:transcript>
       (handles text/html/VTT/SRT/JSON)
    3) Optional fallback: download audio (yt-dlp) + local ASR with faster-whisper (if enabled)
- Summarize: local Ollama
- Write: Obsidian note with YAML front matter (+ consumed: today)

Env (.env):
  OBSIDIAN_VAULT=/vault
  OBSIDIAN_FOLDER=Media/Podcasts
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  OLLAMA_MODEL=llama3.1:8b
  CONSUMED_TZ=America/Detroit
  YTDLP_COOKIES=/vault/.podcast_cookies.txt              (optional)
  PODCAST_ASR_ENABLE=0|1                                 (optional; enables Whisper fallback)
  PODCAST_ASR_MODEL=base                                 (optional; faster-whisper model, e.g. medium)
"""

from __future__ import annotations

import os
import re
import json
import glob
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv
from yt_dlp import YoutubeDL
import feedparser  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# Optional timezone support
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

DEFAULT_LANGS = ["en", "en-US", "en-GB"]
USER_AGENT = "obsidian-podcast-noter/1.0"


# ----------------------------- Utilities -----------------------------

def _now_date_str() -> str:
    tzname = os.getenv("CONSUMED_TZ", "America/Detroit")
    if ZoneInfo:
        return datetime.now(ZoneInfo(tzname)).date().isoformat()
    return datetime.now().date().isoformat()


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:180]


def fmt_hms(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _json_quote(s: str) -> str:
    return json.dumps(str(s))


# ----------------------------- Metadata (yt-dlp) -----------------------------

def fetch_podcast_metadata(url: str) -> Dict[str, Any]:
    """
    Use yt-dlp to extract episode/page info without downloading audio.
    Works on many host pages (Buzzsprout, Transistor, Spotify/Open, Apple page, etc.)
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "extract_flat": False,
        "extractor_retries": 3,
        "forceipv4": True,
    }
    cookies_file = os.getenv("YTDLP_COOKIES")
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # Try to infer show name
    show = info.get("uploader") or info.get("artist") or info.get("album") or info.get("channel")
    title = info.get("title")
    date = info.get("upload_date") or info.get("release_date")  # YYYYMMDD
    duration = info.get("duration")
    webpage_url = info.get("webpage_url") or url
    audio_url = None

    # Enclosure / audio URL if discoverable
    if isinstance(info.get("formats"), list):
        # Prefer audio-only formats
        audio_formats = [f for f in info["formats"] if f.get("acodec") and not f.get("vcodec")]
        if audio_formats:
            audio_formats.sort(key=lambda f: (f.get("abr") or 0, f.get("filesize") or 0), reverse=True)
            audio_url = audio_formats[0].get("url")
    audio_url = audio_url or info.get("url")

    return {
        "episode_title": title,
        "show": show,
        "publish_date": _fmt_yyyymmdd(date),
        "duration": duration,
        "url": webpage_url,
        "audio_url": audio_url,
        "id": info.get("id"),
    }


def _fmt_yyyymmdd(s: Optional[str]) -> Optional[str]:
    if not s or len(s) != 8:
        return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


# ----------------------------- Transcript: webpage scrape -----------------------------

TRANSCRIPT_HINTS = [
    "transcript", "Transcript", "TRANSCRIPT",
    "show-notes", "shownotes", "show_notes",
    "episode-transcript", "episode_transcript",
]

def _html_text(el: Any) -> str:
    for tag in el(["script", "style", "noscript"]):
        tag.decompose()
    text = el.get_text("\n", strip=True)
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def try_scrape_transcript_from_page(url: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Heuristic: fetch the page, look for obvious transcript blocks or links.
    Returns (text, segments|None). Text can be long; segments omitted for page scrape.
    """
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except Exception:
        return None, None

    soup = BeautifulSoup(r.text, "lxml")

    # 1) Obvious transcript containers by id/class
    for hint in TRANSCRIPT_HINTS:
        node = soup.select_one(f'#{hint}, .{hint}, [data-section*="{hint}"]')
        if node:
            txt = _html_text(node)
            if len(txt.split()) > 50:
                return txt, None

    # 2) Section headers containing 'transcript'
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        if "transcript" in (h.get_text(" ", strip=True) or "").lower():
            # collect following siblings until next header
            buf = [h.get_text(" ", strip=True)]
            for sib in h.find_all_next():
                if sib.name and re.fullmatch(r"h[1-6]", sib.name, re.I):
                    break
                if sib.name in ("p", "div", "li"):
                    buf.append(sib.get_text(" ", strip=True))
            txt = "\n\n".join(x for x in buf if x).strip()
            if len(txt.split()) > 50:
                return txt, None

    # 3) Link to a transcript page
    for a in soup.find_all("a", href=True):
        label = (a.get_text(" ", strip=True) or "") + " " + (a.get("aria-label") or "")
        if "transcript" in label.lower():
            turl = requests.compat.urljoin(url, a["href"])
            try:
                r2 = requests.get(turl, timeout=20, headers={"User-Agent": USER_AGENT})
                r2.raise_for_status()
                soup2 = BeautifulSoup(r2.text, "lxml")
                main = soup2.find("main") or soup2.find("article") or soup2.body
                if main:
                    txt = _html_text(main)
                    if len(txt.split()) > 50:
                        return txt, None
            except Exception:
                pass

    return None, None


# ----------------------------- Transcript: RSS + podcast:transcript -----------------------------

def _discover_rss_link(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "lxml")
    for link in soup.find_all("link", attrs={"rel": "alternate"}):
        t = (link.get("type") or "").lower()
        if "rss" in t or "xml" in t:
            href = link.get("href")
            if href:
                return requests.compat.urljoin(url, href)
    return None


def _fetch(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _choose_entry(feed: Any, page_url: str, page_title: Optional[str]) -> Optional[feedparser.FeedParserDict]:
    # Prefer exact link match; fallback: title containment
    for e in feed.entries:
        if _normalize(e.get("link", "")) == _normalize(page_url):
            return e
    if page_title:
        p = _normalize(page_title)
        best = None
        for e in feed.entries:
            t = _normalize(e.get("title", ""))
            if t and (t in p or p in t):
                best = e
                break
        if best:
            return best
    # last resort: first entry
    return feed.entries[0] if feed.entries else None


def _parse_podcast_transcript_tag(item_xml: ET.Element) -> Optional[Tuple[str, str]]:
    """
    Return (url, mimetype) from <podcast:transcript> if present.
    """
    # podcast namespace
    ns = {"podcast": "https://podcastindex.org/namespace/1.0"}
    for el in item_xml.findall(".//podcast:transcript", ns):
        url = el.attrib.get("url")
        typ = el.attrib.get("type", "")
        if url:
            return url, typ
    return None


def _parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    def _parse_ts(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
        else:
            h, m, s = "0", parts[0], parts[1]
        s = s.replace(",", ".")
        return int(h) * 3600 + int(m) * 60 + float(s)

    lines = [ln.rstrip("\n") for ln in vtt_text.splitlines()]
    segs: List[Dict[str, Any]] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip(); i += 1
        if not ln or ln.startswith(("WEBVTT", "NOTE")):
            continue
        if ln.isdigit() and i < n:
            ln = lines[i].strip(); i += 1
        if "-->" not in ln:
            continue
        try:
            ts1, ts2 = [t.strip() for t in ln.split("-->", 1)]
            start = _parse_ts(ts1.split(" ")[0]); end = _parse_ts(ts2.split(" ")[0])
            buf = []
            while i < n and lines[i].strip():
                buf.append(lines[i].strip()); i += 1
            while i < n and not lines[i].strip():
                i += 1
            text = " ".join(buf).strip()
            if text:
                segs.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        except Exception:
            continue
    return segs


def _parse_srt_to_segments(srt_text: str) -> List[Dict[str, Any]]:
    def _to_sec(h: int, m: int, s: int, ms: int) -> float:
        return h * 3600 + m * 60 + s + ms / 1000.0

    segs: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.M)
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if re.fullmatch(r"\d+", lines[0]):
            lines = lines[1:]
        if not lines:
            continue
        if "-->" not in lines[0]:
            continue
        ts = lines[0]
        m = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
        if not m:
            continue
        sh, sm, ss, sms, eh, em, es, ems = map(int, m.groups())
        start = _to_sec(sh, sm, ss, sms)
        end = _to_sec(eh, em, es, ems)
        text = " ".join(lines[1:])
        if text:
            segs.append({"text": text, "start": start, "duration": max(0.0, end - start)})
    return segs


def try_transcript_via_rss(url: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Discover RSS and try to read Podcasting 2.0 transcript for the matching item.
    Returns (text, segments|None).
    """
    rss = _discover_rss_link(url)
    # Also capture page title for entry matching
    page_html = _fetch(url)
    page_title = None
    if page_html:
        soup = BeautifulSoup(page_html, "lxml")
        meta_title = soup.find("meta", attrs={"property": "og:title"})
        page_title = (meta_title.get("content") if meta_title else soup.title.string if soup.title else None)

    if not rss:
        return None, None

    # Pull both parsed and raw XML to access extension tags
    feed = feedparser.parse(rss)
    raw_xml = _fetch(rss)
    if not raw_xml or not feed.entries:
        return None, None

    entry = _choose_entry(feed, url, page_title)
    if not entry:
        return None, None

    # Find matching <item> via guid or link
    root = ET.fromstring(raw_xml)
    channel = root.find("channel")
    item_xml = None
    if channel is not None:
        for it in channel.findall("item"):
            guid_el = it.find("guid")
            link_el = it.find("link")
            title_el = it.find("title")
            guid = guid_el.text.strip() if guid_el is not None and guid_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            if (entry.get("id") and entry.get("id") == guid) or \
               (_normalize(entry.get("link","")) == _normalize(link)) or \
               (_normalize(entry.get("title","")) == _normalize(title)):
                item_xml = it
                break

    if item_xml is None:
        return None, None

    tr = _parse_podcast_transcript_tag(item_xml)
    if not tr:
        # Sometimes transcript is embedded in <content:encoded>
        content_encoded = item_xml.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content_encoded is not None and content_encoded.text:
            soup = BeautifulSoup(content_encoded.text, "lxml")
            txt = soup.get_text("\n", strip=True)
            if len(txt.split()) > 50:
                return txt, None
        return None, None

    tr_url, tr_type = tr
    tr_text = _fetch(tr_url)
    if not tr_text:
        return None, None

    # Handle common types
    t = (tr_type or "").lower()
    if "vtt" in t:
        segs = _parse_vtt_to_segments(tr_text)
        text = " ".join(s["text"] for s in segs if s.get("text"))
        return text, segs
    if "srt" in t or "subrip" in t:
        segs = _parse_srt_to_segments(tr_text)
        text = " ".join(s["text"] for s in segs if s.get("text"))
        return text, segs
    if "json" in t:
        # Very loose assumption: list of {text,start,duration} or {start,end,text}
        try:
            data = json.loads(tr_text)
            segs = []
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict):
                        text = it.get("text") or it.get("body") or ""
                        start = float(it.get("start", it.get("begin", 0)))
                        dur = float(it.get("duration", max(0.0, float(it.get("end", start)) - start)))
                        if text:
                            segs.append({"text": str(text), "start": start, "duration": dur})
            if segs:
                text = " ".join(s["text"] for s in segs if s.get("text"))
                return text, segs
        except Exception:
            pass

    # text/plain or text/html or unknown → extract readable text
    if "html" in t:
        soup = BeautifulSoup(tr_text, "lxml")
        txt = soup.get_text("\n", strip=True)
        if len(txt.split()) > 10:
            return txt, None
    # plain text
    txt = tr_text.strip()
    if len(txt.split()) > 10:
        return txt, None

    return None, None


# ----------------------------- Optional ASR fallback (faster-whisper) -----------------------------

def try_transcript_via_asr(url: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Download audio (yt-dlp) and transcribe locally using faster-whisper.
    Requires: faster-whisper + ffmpeg in your image, and PODCAST_ASR_ENABLE=1.
    """
    if os.getenv("PODCAST_ASR_ENABLE", "0") != "1":
        return None, None
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None, None

    cookies_file = os.getenv("YTDLP_COOKIES")
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "extractor_retries": 3,
        "forceipv4": True,
        "format": "bestaudio/best",
        "outtmpl": "%(id)s.%(ext)s",
    }
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    tmpdir = tempfile.mkdtemp(prefix="pod_asr_")
    audio_path = None
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # choose the downloaded file
            candidates = glob.glob(os.path.join(os.getcwd(), f"{info.get('id')}.*"))
            if candidates:
                audio_path = candidates[0]
                # move into tmpdir for cleanup
                new_path = os.path.join(tmpdir, os.path.basename(audio_path))
                shutil.move(audio_path, new_path)
                audio_path = new_path

        if not audio_path:
            return None, None

        model_name = os.getenv("PODCAST_ASR_MODEL", "base")
        model = WhisperModel(model_name, compute_type="int8")  # lightweight default
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        segs_list: List[Dict[str, Any]] = []
        for seg in segments:
            segs_list.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "duration": float(seg.end - seg.start)
            })
        text = " ".join(s["text"] for s in segs_list if s.get("text"))
        return text, segs_list if segs_list else None
    except Exception:
        return None, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------- Summarization (Ollama) -----------------------------

def chunk_text_by_chars(text: str, max_chars: int = 15000) -> List[str]:
    if len(text) <= max_chars: return [text]
    chunks, buf, total = [], [], 0
    for token in text.split():
        if total + len(token) + 1 > max_chars:
            chunks.append(" ".join(buf)); buf, total = [token], len(token)
        else:
            buf.append(token); total += len(token) + 1
    if buf: chunks.append(" ".join(buf))
    return chunks


def _check_ollama(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        return r.status_code == 200 and r.headers.get("content-type","").startswith("application/json")
    except Exception:
        return False


def call_ollama_any(base_url: str, model: str, prompt: str) -> str:
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 200:
            return r.json().get("response","")
    except Exception:
        pass
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": [{"role":"user","content": prompt}], "stream": False},
            timeout=600,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 200:
            j = r.json()
            if "message" in j and isinstance(j["message"], dict):
                return j["message"].get("content","")
            if "response" in j:
                return j["response"]
            if "choices" in j and j["choices"]:
                return j["choices"][0].get("message",{}).get("content","")
    except Exception:
        pass
    raise RuntimeError(f"Ollama API not responding at {base_url}.")


def summarize_with_ollama(base_url: str, model: str, title: str, show: str, url: str, transcript: str, map_reduce: bool=True) -> str:
    if not _check_ollama(base_url):
        raise RuntimeError(f"{base_url} does not look like an Ollama server (/api/tags not OK).")

    SYS = (
        "You are a precise note-taker creating concise, accurate summaries for Obsidian. "
        "Prefer structured Markdown with headings, bullet points, and short quotes. "
        "Base everything only on the transcript; do not invent facts."
    )

    def map_prompt(chunk: str) -> str:
        return (
            f"{SYS}\n\nYou will summarize part of a podcast transcript.\n"
            f"Show: {show}\nTitle: {title}\nURL: {url}\n\n"
            f"Write concise bullets with key ideas, facts, steps, and short quotes.\n\n"
            f'Transcript chunk:\n"""' + chunk + '"""'
        )

    def reduce_prompt(parts_md: str) -> str:
        return (
            f"{SYS}\n\nUnify the partial summaries into a well-structured Markdown note:\n\n{parts_md}\n\n"
            "# Summary\n"
            "1–3 paragraph executive summary.\n\n"
            "## Key Points\n- Bulleted list of the most important takeaways.\n\n"
            "## Details & Timestamps\n- Group related bullets; include timestamps when available.\n\n"
            "## Action Items / How-To (if applicable)\n- Steps or recommendations.\n\n"
            "## Memorable Quotes\n- Short quotes (≤20 words) with timestamps."
        )

    if (not map_reduce) or len(transcript) < 15000:
        return call_ollama_any(base_url, model, map_prompt(transcript))

    parts = [call_ollama_any(base_url, model, map_prompt(ch)) for ch in chunk_text_by_chars(transcript, 15000)]
    merged = "\n\n---\n\n".join(parts)
    return call_ollama_any(base_url, model, reduce_prompt(merged))


# ----------------------------- YAML + Note Writing -----------------------------

def yaml_front_matter(meta: Dict[str, Any]) -> str:
    props: Dict[str, Any] = {
        "title": meta.get("episode_title"),
        "type": "podcast",
        "show": meta.get("show"),
        "episode_id": meta.get("id"),
        "url": meta.get("url"),
        "audio_url": meta.get("audio_url"),
        "publish_date": meta.get("publish_date"),
        "duration_seconds": meta.get("duration"),
        "tags": ["media", "podcast"],
        "consumed": _now_date_str(),
    }
    lines = ["---"]
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {_json_quote(item)}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {_json_quote(v)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def make_metadata_only_body(meta: Dict[str, Any], transcript_text: Optional[str]=None) -> str:
    dur = fmt_hms(meta.get("duration"))
    lines: List[str] = []
    lines.append("# Summary")
    lines.append("_(no summary generated)_")
    lines.append("")
    lines.append("## Episode")
    lines.append(f"- **Show:** {meta.get('show') or 'Unknown Show'}")
    lines.append(f"- **Title:** {meta.get('episode_title') or 'Untitled'}")
    if meta.get("publish_date"): lines.append(f"- **Published:** {meta.get('publish_date')}")
    if dur: lines.append(f"- **Duration:** {dur}")
    if meta.get("url"): lines.append(f"- **Page:** {meta.get('url')}")
    if meta.get("audio_url"): lines.append(f"- **Audio:** {meta.get('audio_url')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("_Add your notes here…_")
    lines.append("")
    if transcript_text:
        lines.append("<details>")
        lines.append("<summary><strong>Transcript</strong></summary>\n")
        lines.append(transcript_text.strip())
        lines.append("\n</details>\n")
    return "\n".join(lines)


def write_obsidian_note(vault_path: Path, folder: str, meta: Dict[str, Any], body_md: str) -> Path:
    show = meta.get("show") or "Podcast"
    title = meta.get("episode_title") or "Untitled Episode"
    date = meta.get("publish_date") or ""
    filename = sanitize_filename(f"{show} - {title} (Podcast) ({date}).md" if date else f"{show} - {title} (Podcast).md")

    target_dir = vault_path / folder if folder else vault_path
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / filename
    if path.exists():
        stem, suffix, i = path.stem, path.suffix, 2
        while (target_dir / f"{stem} #{i}{suffix}").exists():
            i += 1
        path = target_dir / f"{stem} #{i}{suffix}"

    front = yaml_front_matter(meta)
    content = f"{front}\n{body_md.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return path


# ----------------------------- Orchestrator -----------------------------

def process_podcast(
    podcast_url: str,
    *,
    vault: Optional[str] = None,
    folder: Optional[str] = "",
    langs: Optional[List[str]] = None,
    ollama_base: Optional[str] = None,
    model: Optional[str] = None,
    map_reduce: bool = True,
    no_summary: bool = False,
    include_transcript: bool = False,
) -> Dict[str, Any]:
    """
    Main entrypoint (mirrors youtube_notes.process_youtube signature).
    Returns:
      { "meta": {...}, "summary": "<md>", "note_path": "/abs/path.md" }
    Raises on hard failures (so RQ can record them).
    """
    load_dotenv()

    vault = vault or os.getenv("OBSIDIAN_VAULT")
    folder = folder if folder is not None else (os.getenv("OBSIDIAN_FOLDER") or "")
    langs = langs or [s.strip() for s in (os.getenv("PODCAST_LANGS") or os.getenv("YT_LANGS","")).split(",") if s.strip()] or DEFAULT_LANGS
    ollama_base = ollama_base or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # 1) Metadata
    meta = fetch_podcast_metadata(podcast_url)

    # 2) Transcript (page → RSS → ASR)
    page_txt, _ = try_scrape_transcript_from_page(podcast_url)
    rss_txt, rss_segs = (None, None) if page_txt else try_transcript_via_rss(podcast_url)
    asr_txt, asr_segs = (None, None) if (page_txt or rss_txt) else try_transcript_via_asr(podcast_url)

    transcript_text = page_txt or rss_txt or asr_txt
    transcript_segments = rss_segs or asr_segs  # only used if we care about timestamps later

    # 3) Body
    if no_summary:
        body = make_metadata_only_body(meta, transcript_text if include_transcript else None)
    else:
        if not transcript_text:
            raise RuntimeError("No transcript found (page & RSS), and ASR fallback disabled or failed.")
        body = summarize_with_ollama(
            base_url=ollama_base,
            model=model,
            title=meta.get("episode_title") or "",
            show=meta.get("show") or "",
            url=meta.get("url") or podcast_url,
            transcript=transcript_text,
            map_reduce=map_reduce,
        )

    # 4) Write note
    note_path = None
    if vault:
        note_path = str(write_obsidian_note(Path(vault).expanduser().resolve(), folder or "", meta, body))

    return {"meta": meta, "summary": body, "note_path": note_path}
