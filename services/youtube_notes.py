# services/youtube_notes.py
"""
YouTube → Obsidian utilities (Docker/RQ friendly)

Features
- Extract video ID from a YouTube URL
- Fetch metadata via yt-dlp (no media download)
- Get transcript via youtube-transcript-api (instance API: fetch/list)
  * If no transcript available, fall back to yt-dlp auto/manual subtitles (VTT) and parse them
  * Optional cookie support for age/consent-gated videos via YTDLP_COOKIES
- Summarize transcript using a local Ollama server (generate/chat fallback)
- Write a Markdown note with YAML front matter into an Obsidian vault
- Single entrypoint: process_youtube(...)

Environment (.env)
- OBSIDIAN_VAULT=/vault            # inside container mount point
- OBSIDIAN_FOLDER=Media/YouTube
- YT_LANGS=en,en-US,en-GB
- OLLAMA_BASE_URL=http://host.docker.internal:11434
- OLLAMA_MODEL=llama3.1:8b
- YTDLP_COOKIES=/vault/.youtube_cookies.txt   # optional Netscape-format cookies file
"""
from datetime import datetime
import os
import re
import json
import glob
import shutil
import tempfile
import random
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from yt_dlp import YoutubeDL

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

DEFAULT_LANGS = ["en", "en-US", "en-GB"]
USER_AGENT = "obsidian-youtube-noter/2.1"

# Default prompts for Ollama summarization (can be overridden via .env)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
    "You are a precise note-taker creating concise, accurate summaries for Obsidian. "
    "Prefer structured Markdown with headings, bullet points, and short quotes. "
    "Base everything only on the transcript; do not invent facts."
)

DEFAULT_SUMMARY_PROMPT = os.getenv(
    "YOUTUBE_SUMMARY_PROMPT",
    "You will summarize a YouTube transcript.\n"
    "Title: {title}\nURL: {url}\n\n"
    "Provide a well-structured Markdown summary with:\n"
    "# Summary\n1–3 paragraph executive summary.\n\n"
    "## Key Points\n- Bulleted list of the most important takeaways.\n\n"
    "## Details & Timestamps\n- Group related bullets; include timestamps when available.\n\n"
    "## Action Items / How-To (if applicable)\n- Steps or recommendations.\n\n"
    "## Memorable Quotes\n- Short quotes (≤20 words) with timestamps.\n\n"
    'Transcript:\n"""{transcript}"""'
)

DEFAULT_SEGMENT_PROMPT = os.getenv(
    "YOUTUBE_SEGMENT_PROMPT",
    "Summarize this portion of a YouTube stream/video.\n"
    "Title: {title}\nURL: {url}\n"
    "Time Range: {start_hms} - {end_hms}\n\n"
    "Provide a concise summary with:\n"
    "- Main topics discussed\n"
    "- Key points and takeaways\n"
    "- Notable quotes (if any)\n\n"
    'Transcript:\n"""{transcript}"""'
)


# ----------------------------- URL & metadata -----------------------------

def _get_random_proxy() -> Optional[str]:
    """Read proxy list file and return a random proxy."""
    proxy_file = os.getenv("YTDLP_PROXY_FILE")
    if not proxy_file or not os.path.exists(proxy_file):
        return None
    
    try:
        with open(proxy_file, "r", encoding="utf-8") as f:
            proxies = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        
        if proxies:
            return random.choice(proxies)
    except Exception:
        pass
    
    return None

def extract_yt_id(url: str) -> Optional[str]:
    """Extract the 11-char YouTube video ID from various URL styles or accept a bare ID."""
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"(?:youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


def fetch_metadata(url: str, use_cookies: bool = False, use_proxy: bool = False) -> Tuple[Dict[str, Any], Optional[str]]:
    """Use yt-dlp to get basic metadata without downloading the media.
    Returns: (metadata_dict, proxy_used)
    """
    proxy_used = None
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "extract_flat": False,
        "extractor_retries": 3,
        "forceipv4": True,
        "js_runtimes": {"bun": {"path": "/root/.bun/bin/bun"}},
       # "extractor_args": {"youtube": {"player_client": ["web"]}},
    }
    if use_cookies:
        cookies_file = os.getenv("YTDLP_COOKIES")
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts["cookiefile"] = cookies_file
    if use_proxy:
        proxy = _get_random_proxy()
        if proxy:
            ydl_opts["proxy"] = proxy
            proxy_used = proxy
    
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "duration": info.get("duration"),        # seconds
        "url": info.get("webpage_url") or url,
        "id": info.get("id"),
        "channel_id": info.get("channel_id"),
    }, proxy_used


# ----------------------------- VTT parsing + yt-dlp fallback -----------------------------

def _parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    """
    Minimal VTT parser -> list of {text, start, duration}.
    Good enough for summarization (ignores styling and notes).
    """
    def _parse_ts(ts: str) -> float:
        # "HH:MM:SS.mmm" or "MM:SS.mmm"
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
        ln = lines[i].strip()
        i += 1
        if not ln or ln.startswith(("WEBVTT", "NOTE")):
            continue
        # Optional numeric cue id
        if ln.isdigit() and i < n:
            ln = lines[i].strip()
            i += 1
        if "-->" not in ln:
            continue
        try:
            ts1, ts2 = [t.strip() for t in ln.split("-->", 1)]
            start = _parse_ts(ts1.split(" ")[0])
            end = _parse_ts(ts2.split(" ")[0])
            buf: List[str] = []
            # Gather cue text until blank line
            while i < n and lines[i].strip():
                buf.append(lines[i].strip())
                i += 1
            # Skip trailing blanks
            while i < n and not lines[i].strip():
                i += 1
            text = " ".join(buf).strip()
            if text:
                segs.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        except Exception:
            continue
    return segs


def _fetch_subtitles_via_ytdlp(youtube_url: str, langs: Optional[List[str]] = None, use_cookies: bool = False, use_proxy: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ask yt-dlp to download auto or manual subtitles (VTT) for the given URL, then parse them.
    Returns (full_text, segments). Raises if nothing is available.
    Uses cookies if use_cookies=True and YTDLP_COOKIES (Netscape format) is set in env.
    Uses proxy if use_proxy=True and YTDLP_PROXY is set in env.
    """
    tmpdir = tempfile.mkdtemp(prefix="subs_")
    try:
        lang_list = [l for l in langs if l] or ["en", "en-US", "en-GB"]
        # Ensure common English fallbacks are present, preserve order/dedup
        lang_list = list(dict.fromkeys(lang_list + ["en", "en-US", "en-GB"]))

        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitlesformat": "vtt",
            "subtitleslangs": lang_list,
            "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
            "nocheckcertificate": True,
            "extractor_retries": 3,
            "forceipv4": True,
            "js_runtimes": {"bun": {"path": "/root/.bun/bin/bun"}},
           # "extractor_args": {"youtube": {"player_client": ["web"]}},
        }
        if use_cookies:
            cookies_file = os.getenv("YTDLP_COOKIES")
            if cookies_file and os.path.exists(cookies_file):
                ydl_opts["cookiefile"] = cookies_file
        if use_proxy:
            proxy = _get_random_proxy()
            if proxy:
                ydl_opts["proxy"] = proxy

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            vid = info.get("id")

        # Prefer requested languages, then any VTT we got
        candidates: List[str] = []
        for code in lang_list:
            candidates += glob.glob(os.path.join(tmpdir, f"{vid}.{code}.vtt"))
        if not candidates:
            candidates = glob.glob(os.path.join(tmpdir, f"{vid}.*.vtt"))
        if not candidates:
            raise RuntimeError("No VTT subtitles downloaded")

        with open(candidates[0], "r", encoding="utf-8", errors="ignore") as f:
            segs = _parse_vtt_to_segments(f.read())
        if not segs:
            raise RuntimeError("Downloaded VTT was empty or unparseable")
        text = " ".join(s["text"] for s in segs if s.get("text"))
        return text, segs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------- Instance API transcript fetch -----------------------------

def _concat_text(segs: List[Dict[str, Any]]) -> str:
    return " ".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())


def _to_raw(obj: Any) -> List[Dict[str, Any]]:
    """Normalize FetchedTranscript/list to a list[dict] with keys: text, start, duration."""
    if hasattr(obj, "to_raw_data"):  # FetchedTranscript (new API)
        return obj.to_raw_data()  # type: ignore[no-any-return]
    if isinstance(obj, list):
        return obj  # type: ignore[no-any-return]
    try:
        return list(obj)  # type: ignore[no-any-return]
    except Exception:
        return []


def try_get_transcript(video_id: str, langs: List[str], youtube_url: Optional[str] = None, use_cookies: bool = False, use_proxy: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fetch transcript via NEW youtube-transcript-api instance API (.fetch/.list).
    If unavailable, fall back to yt-dlp auto/manual VTT subtitles.

    Returns (full_text, segments_list_of_dicts).
    """
    if not youtube_url:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    ytt = YouTubeTranscriptApi()

    # 1) Direct fetch with preferred languages
    try:
        ft = ytt.fetch(video_id, languages=langs)
        segs = _to_raw(ft)
        if segs:
            return _concat_text(segs), segs
    except (NoTranscriptFound, TranscriptsDisabled):
        pass
    except Exception:
        pass

    # 2) Enumerate transcripts and pick the best
    try:
        tlist = ytt.list(video_id)  # TranscriptList
    except Exception:
        tlist = None

    if tlist is not None:
        langs_nonempty = [l for l in langs if l] or DEFAULT_LANGS

        # Prefer generic finder if present
        finder = getattr(tlist, "find_transcript", None)
        if callable(finder):
            try:
                tr = finder(langs_nonempty)
                segs = _to_raw(tr.fetch())
                if segs:
                    return _concat_text(segs), segs
            except Exception:
                pass

        def _all_langs() -> List[str]:
            try:
                return [t.language_code for t in tlist]
            except Exception:
                return []

        for fname, pool in [
            ("find_manually_created_transcript", langs_nonempty),
            ("find_manually_created_transcript", _all_langs()),
            ("find_generated_transcript",        langs_nonempty),
            ("find_generated_transcript",        _all_langs()),
        ]:
            f = getattr(tlist, fname, None)
            if callable(f) and pool:
                try:
                    tr = f(pool)
                    segs = _to_raw(tr.fetch())
                    if segs:
                        return _concat_text(segs), segs
                except Exception:
                    continue

    # 3) Final fallback: download auto/manual VTT subs via yt-dlp
    logger.info("Attempting to fetch VTT subtitles via yt-dlp...")
    return _fetch_subtitles_via_ytdlp(youtube_url, langs, use_cookies, use_proxy)


# ----------------------------- Summarization (Ollama) -----------------------------

def chunk_text_by_chars(text: str, max_chars: int = 15000) -> List[str]:
    """Greedy word-based chunking to keep prompts under a safe size for local models."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    buf: List[str] = []
    total = 0
    for token in text.split():
        if total + len(token) + 1 > max_chars:
            chunks.append(" ".join(buf))
            buf, total = [token], len(token)
        else:
            buf.append(token)
            total += len(token) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_segments_by_duration(segments: List[Dict[str, Any]], duration_seconds: int = 3600) -> List[Tuple[str, float, float]]:
    """Chunk transcript segments by time duration (e.g., per hour).
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
        return r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json")
    except Exception:
        return False


def call_ollama_any(base_url: str, model: str, prompt: str, context_length: int = 4096) -> str:
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
            j = r.json()
            return j.get("response", "")
    except Exception:
        pass

    # /api/chat
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model, 
                "messages": [{"role": "user", "content": prompt}], 
                "stream": False,
                "options": {"num_ctx": context_length}
            },
            timeout=600,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict):
                if "message" in j and isinstance(j["message"], dict):
                    return j["message"].get("content", "")
                if "response" in j:
                    return j["response"]
                if "choices" in j and j["choices"]:
                    return j["choices"][0].get("message", {}).get("content", "")
    except Exception:
        pass

    raise RuntimeError(
        f"Ollama API not responding at {base_url}. "
        "Make sure /api/tags returns JSON and that your reverse proxy forwards /api/*."
    )


def ollama_summarize(
    base_url: str,
    model: str,
    title: str,
    url: str,
    transcript: str,
    context_length: int = 4096,
    segments: Optional[List[Dict[str, Any]]] = None,
    per_hour: bool = False,
    segment_duration: int = 3600,
    system_prompt: Optional[str] = None,
    summary_prompt: Optional[str] = None,
    segment_prompt: Optional[str] = None,
) -> str:
    """Summarize the transcript with a local Ollama model.
    
    Args:
        per_hour: If True, create independent summaries for each time segment.
                 Requires segments parameter.
        segment_duration: Duration in seconds for each segment (default: 3600 = 1 hour).
        system_prompt: Optional system prompt (prepended to all prompts).
        summary_prompt: Optional full summary prompt template.
        segment_prompt: Optional segment prompt template.
    """
    if not _check_ollama(base_url):
        raise RuntimeError(f"{base_url} does not look like an Ollama server (/api/tags not OK).")

    # Use provided prompts or defaults
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    sum_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
    seg_prompt = segment_prompt or DEFAULT_SEGMENT_PROMPT

    def full_prompt(text: str) -> str:
        # Use summary prompt with system prompt prepended
        return (
            sys_prompt + "\n\n" +
            sum_prompt.format(title=title, url=url, transcript=text)
        )

    # Per-hour mode: independent summaries for each time segment
    if per_hour and segments:
        time_chunks = chunk_segments_by_duration(segments, duration_seconds=segment_duration)
        if not time_chunks:
            return "# Summary\n\n_(No content to summarize)_"
        
        result_parts: List[str] = []
        for i, (chunk_text, start_sec, end_sec) in enumerate(time_chunks, 1):
            start_hms = fmt_hms(int(start_sec))
            end_hms = fmt_hms(int(end_sec))
            
            # Use segment prompt with system prompt prepended
            segment_prompt_text = (
                sys_prompt + "\n\n" +
                seg_prompt.format(
                    title=title, url=url, start_hms=start_hms, end_hms=end_hms, transcript=chunk_text
                )
            )
            
            summary = call_ollama_any(base_url, model, segment_prompt_text, context_length)
            result_parts.append(f"## Segment {i}: {start_hms} - {end_hms}\n\n{summary}")
        
        return "# Summary\n\n" + "\n\n".join(result_parts)

    # Default: full transcript summary
    return call_ollama_any(base_url, model, full_prompt(transcript), context_length)


# ----------------------------- Obsidian note -----------------------------

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:180]


def fmt_upload_date(s: Optional[str]) -> Optional[str]:
    if not s or len(s) != 8:
        return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def yaml_front_matter(meta: Dict[str, Any]) -> str:
    """
    Produce a YAML block. Use json.dumps for safe string quoting (YAML superset of JSON).
    """
    consumed_str = datetime.now().date().isoformat()

    props: Dict[str, Any] = {
        "title": meta.get("title"),
        "type": "youtube",
        "youtube_id": meta.get("id"),
        "url": meta.get("url"),
        "channel": meta.get("channel"),
        "upload_date": fmt_upload_date(meta.get("upload_date")),
        "duration_seconds": meta.get("duration"),
        "tags": ["media", "youtube"],
        "consumed": consumed_str
    }
    lines = ["---"]
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {json.dumps(str(item))}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {json.dumps(str(v))}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def fmt_hms(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def make_metadata_only_body(meta: Dict[str, Any], transcript_text: Optional[str] = None) -> str:
    dur = fmt_hms(meta.get("duration"))
    upload = fmt_upload_date(meta.get("upload_date"))
    url = meta.get("url")
    channel = meta.get("channel")

    lines: List[str] = []
    lines.append("# Summary")
    lines.append("_(no summary generated)_")
    lines.append("")
    lines.append("## Video")
    lines.append(f"- **Title:** {meta.get('title') or 'Untitled'}")
    if channel:
        lines.append(f"- **Channel:** {channel}")
    if upload:
        lines.append(f"- **Uploaded:** {upload}")
    if dur:
        lines.append(f"- **Duration:** {dur}")
    if url:
        lines.append(f"- **URL:** {url}")
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
    title = meta.get("title") or "Untitled Video"
    date = fmt_upload_date(meta.get("upload_date")) or ""
    filename = sanitize_filename(f"{title} - YouTube ({date}).md" if date else f"{title} - YouTube.md")

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

def process_youtube(
    youtube_url: str,
    *,
    vault: Optional[str] = None,
    folder: Optional[str] = "",
    langs: Optional[List[str]] = None,
    ollama_base: Optional[str] = None,
    model: Optional[str] = None,
    no_summary: bool = False,
    include_transcript: bool = False,
    use_cookies: bool = False,
    use_proxy: bool = False,
    context_length: int = 4096,
    per_hour: bool = False,
    segment_duration: int = 3600,
    system_prompt: Optional[str] = None,
    summary_prompt: Optional[str] = None,
    segment_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entrypoint used by Flask/RQ.

    Returns:
      {
        "meta": {...},
        "summary": "<markdown body>",
        "note_path": "/abs/path/to/note.md"  # if vault provided
      }
    Raises on hard failures (so RQ can record them).
    """
    load_dotenv()

    # Resolve env defaults
    vault = vault or os.getenv("OBSIDIAN_VAULT")
    folder = folder if folder is not None else (os.getenv("OBSIDIAN_FOLDER") or "")
    langs = langs or [s.strip() for s in (os.getenv("YT_LANGS") or "").split(",") if s.strip()] or DEFAULT_LANGS
    ollama_base = ollama_base or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    vid = extract_yt_id(youtube_url)
    if not vid:
        raise ValueError("Invalid YouTube URL/ID")

    logger.info(f"Starting YouTube processing for: {youtube_url} (ID: {vid})")
    logger.info(f"Vault: {vault}, Folder: {folder}, No summary: {no_summary}")
    logger.info(f"Use cookies: {use_cookies}, Use proxy: {use_proxy}")

    # Metadata first (always)
    try:
        meta, proxy_used = fetch_metadata(youtube_url, use_cookies=use_cookies, use_proxy=use_proxy)
        logger.info(f"Metadata fetched successfully. Title: {meta.get('title')}")
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {str(e)}")
        raise

    # Build note body
    if no_summary:
        logger.info("Skipping summary generation (no_summary=True)")
        transcript_text: Optional[str] = None
        if include_transcript:
            try:
                logger.info("Fetching transcript for inclusion...")
                transcript_text, _ = try_get_transcript(vid, langs, youtube_url=youtube_url, use_cookies=use_cookies, use_proxy=use_proxy)
            except Exception as e:
                logger.warning(f"Failed to fetch transcript: {str(e)}")
                transcript_text = None
        body = make_metadata_only_body(meta, transcript_text=transcript_text)
    else:
        logger.info("Fetching transcript for summarization...")
        try:
            transcript_text, segments = try_get_transcript(vid, langs, youtube_url=youtube_url, use_cookies=use_cookies, use_proxy=use_proxy)
            logger.info(f"Transcript fetched. Length: {len(transcript_text)} chars, Segments: {len(segments) if segments else 0}")
        except Exception as e:
            logger.error(f"Failed to get transcript: {str(e)}")
            raise
        
        logger.info("Starting summarization...")
        try:
            body = ollama_summarize(
                base_url=ollama_base,
                model=model,
                title=meta.get("title") or "",
                url=meta.get("url") or youtube_url,
                transcript=transcript_text,
                context_length=context_length,
                segments=segments,
                per_hour=per_hour,
                segment_duration=segment_duration,
                system_prompt=system_prompt,
                summary_prompt=summary_prompt,
                segment_prompt=segment_prompt,
            )
            logger.info("Summarization completed successfully")
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise

    note_path: Optional[str] = None
    if vault:
        logger.info(f"Writing Obsidian note to vault: {vault}")
        try:
            note_path = str(write_obsidian_note(Path(vault).expanduser().resolve(), folder or "", meta, body))
            logger.info(f"Note written successfully to: {note_path}")
        except Exception as e:
            logger.error(f"Failed to write note: {str(e)}")
            raise

    logger.info("YouTube processing completed successfully")
    return {"meta": meta, "summary": body, "note_path": note_path, "proxy_used": proxy_used}
