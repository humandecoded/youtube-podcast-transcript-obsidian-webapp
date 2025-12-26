# services/podcast_notes.py
"""
Podcast → Obsidian utilities

- Input: podcast episode URL (Apple/Spotify/site/host page or RSS <item> link)
- Metadata: via yt-dlp (no media download)
- Transcript: download audio (yt-dlp) + local ASR with faster-whisper
- Summarize: local Ollama
- Write: Obsidian note with YAML front matter (+ consumed: today)

Env (.env):
  OBSIDIAN_VAULT=/vault
  OBSIDIAN_FOLDER=Media/Podcasts
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  OLLAMA_MODEL=llama3.1:8b
  CONSUMED_TZ=America/Detroit
  YTDLP_COOKIES=/vault/.podcast_cookies.txt              (optional)
  PODCAST_ASR_ENABLE=1                                   (required; enables Whisper transcription)
  PODCAST_ASR_MODEL=base                                 (optional; faster-whisper model, e.g. medium, large-v3)
  PODCAST_ASR_DEVICE=cpu|cuda                            (optional; default: cpu)
  PODCAST_ASR_COMPUTE=int8|float16|float32               (optional; default: int8)
  PODCAST_ASR_BATCH_SIZE=1                               (optional; batch size for transcription, higher=faster on GPU)
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

        model_name = os.getenv("PODCAST_ASR_MODEL")
        device = os.getenv("PODCAST_ASR_DEVICE")
        compute_type = os.getenv("PODCAST_ASR_COMPUTE")
        batch_size = int(os.getenv("PODCAST_ASR_BATCH_SIZE", "1"))
        
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(
            audio_path, 
            vad_filter=True,
            batch_size=batch_size
        )
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


def chunk_segments_by_duration(segments: List[Dict[str, Any]], duration_seconds: int = 1800) -> List[Tuple[str, float, float]]:
    """Chunk transcript segments by time duration (e.g., per 30 minutes).
    Returns: [(text, start_time, end_time), ...]
    """
    if not segments:
        return []
    
    chunks: List[Tuple[str, float, float]] = []
    current_texts: List[str] = []
    chunk_start = 0.0
    current_end = duration_seconds
    
    for seg in segments:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        
        if not text:
            continue
            
        # If this segment starts beyond current chunk boundary
        if start >= current_end:
            # Save current chunk
            if current_texts:
                chunks.append((" ".join(current_texts), chunk_start, current_end))
            # Start new chunk
            chunk_start = current_end
            current_end = chunk_start + duration_seconds
            current_texts = [text]
        else:
            current_texts.append(text)
    
    # Save final chunk
    if current_texts:
        # Use actual last segment time if available
        actual_end = segments[-1].get("start", current_end) + segments[-1].get("duration", 0)
        chunks.append((" ".join(current_texts), chunk_start, min(actual_end, current_end)))
    
    return chunks


def _check_ollama(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        return r.status_code == 200 and r.headers.get("content-type","").startswith("application/json")
    except Exception:
        return False


def call_ollama_any(base_url: str, model: str, prompt: str, context_length: int = 15000) -> str:
    """Try /api/generate, fallback to /api/chat, handle a few proxy shapes."""
    # /api/generate
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model, 
                "prompt": prompt, 
                "stream": False,
                "options": {"num_ctx": context_length}
            },
            timeout=600,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 200:
            return r.json().get("response","")
    except Exception:
        pass
    # /api/chat
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model, 
                "messages": [{"role":"user","content": prompt}], 
                "stream": False,
                "options": {"num_ctx": context_length}
            },
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


def summarize_with_ollama(base_url: str, model: str, title: str, show: str, url: str, transcript: str, segments: Optional[List[Dict[str, Any]]] = None, map_reduce: bool=True, chunk_size: int=15000, context_length: int=15000, per_segment: bool=False, segment_duration: int=1800) -> str:
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

    # Per-segment mode: independent summaries without reduce phase
    if per_segment and segments:
        time_chunks = chunk_segments_by_duration(segments, duration_seconds=segment_duration)
        if not time_chunks:
            return "# Summary\n\n_(No content to summarize)_"
        
        result_parts: List[str] = []
        for i, (chunk_text, start_sec, end_sec) in enumerate(time_chunks, 1):
            start_hms = fmt_hms(int(start_sec))
            end_hms = fmt_hms(int(end_sec))
            
            segment_prompt = (
                f"{SYS}\n\nSummarize this portion of a podcast episode.\n"
                f"Show: {show}\nTitle: {title}\nURL: {url}\n"
                f"Time Range: {start_hms} - {end_hms}\n\n"
                f"Provide a concise summary with:\n"
                f"- Main topics discussed\n"
                f"- Key points and takeaways\n"
                f"- Notable quotes (if any)\n\n"
                f'Transcript:\n"""' + chunk_text + '"""'
            )
            
            summary = call_ollama_any(base_url, model, segment_prompt, context_length)
            result_parts.append(f"## Segment {i}: {start_hms} - {end_hms}\n\n{summary}")
        
        return "# Summary\n\n" + "\n\n".join(result_parts)

    if (not map_reduce) or len(transcript) < chunk_size:
        return call_ollama_any(base_url, model, map_prompt(transcript), context_length)

    parts = [call_ollama_any(base_url, model, map_prompt(ch), context_length) for ch in chunk_text_by_chars(transcript, chunk_size)]
    merged = "\n\n---\n\n".join(parts)
    return call_ollama_any(base_url, model, reduce_prompt(merged), context_length)


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
    chunk_size: int = 15000,
    context_length: int = 15000,
    per_segment: bool = False,
    segment_duration: int = 1800,
) -> Dict[str, Any]:
    """
    Main entrypoint (mirrors youtube_notes.process_youtube signature).
    Returns:
      { "meta": {...}, "summary": "<md>", "note_path": "/abs/path.md" }
    Raises on hard failures (so RQ can record them).
    chunk_size: Character-based chunking size for map-reduce.
    context_length: Ollama context window size (num_ctx).
    per_segment: If True, create independent summaries without reduce phase.
    segment_duration: Duration in seconds for each segment (default: 1800 = 30 minutes).
    """
    load_dotenv()

    vault = vault or os.getenv("OBSIDIAN_VAULT")
    folder = folder if folder is not None else (os.getenv("OBSIDIAN_FOLDER") or "")
    langs = langs or [s.strip() for s in (os.getenv("PODCAST_LANGS") or os.getenv("YT_LANGS","")).split(",") if s.strip()] or DEFAULT_LANGS
    ollama_base = ollama_base or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # 1) Metadata
    meta = fetch_podcast_metadata(podcast_url)

    # 2) Transcript (ASR only - download and transcribe audio)
    transcript_text, transcript_segments = try_transcript_via_asr(podcast_url)

    # 3) Body
    if no_summary:
        body = make_metadata_only_body(meta, transcript_text if include_transcript else None)
    else:
        if not transcript_text:
            raise RuntimeError("Audio transcription failed. Ensure PODCAST_ASR_ENABLE=1 and faster-whisper is installed.")
        body = summarize_with_ollama(
            base_url=ollama_base,
            model=model,
            title=meta.get("episode_title") or "",
            show=meta.get("show") or "",
            url=meta.get("url") or podcast_url,
            transcript=transcript_text,
            segments=transcript_segments,
            map_reduce=map_reduce,
            chunk_size=chunk_size,
            context_length=context_length,
            per_segment=per_segment,
            segment_duration=segment_duration,
        )

    # 4) Write note
    note_path = None
    if vault:
        note_path = str(write_obsidian_note(Path(vault).expanduser().resolve(), folder or "", meta, body))

    return {"meta": meta, "summary": body, "note_path": note_path}
