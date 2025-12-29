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
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

# Optional timezone support
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

DEFAULT_LANGS = ["en", "en-US", "en-GB"]
USER_AGENT = "obsidian-podcast-noter/1.0"

# Default prompts for Ollama summarization (can be overridden via .env)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
    "You are a precise note-taker creating concise, accurate summaries for Obsidian. "
    "Prefer structured Markdown with headings, bullet points, and short quotes. "
    "Base everything only on the transcript; do not invent facts."
)

DEFAULT_PODCAST_SUMMARY_PROMPT = os.getenv(
    "PODCAST_SUMMARY_PROMPT",
    "You will summarize a podcast transcript.\n"
    "Show: {show}\nTitle: {title}\nURL: {url}\n\n"
    "Provide a well-structured Markdown summary with:\n"
    "# Summary\n1–3 paragraph executive summary.\n\n"
    "## Key Points\n- Bulleted list of the most important takeaways.\n\n"
    "## Details & Timestamps\n- Group related bullets; include timestamps when available.\n\n"
    "## Action Items / How-To (if applicable)\n- Steps or recommendations.\n\n"
    "## Memorable Quotes\n- Short quotes (≤20 words) with timestamps.\n\n"
    'Transcript:\n"""{transcript}"""'
)

DEFAULT_PODCAST_SEGMENT_PROMPT = os.getenv(
    "PODCAST_SEGMENT_PROMPT",
    "Summarize this portion of a podcast episode.\n"
    "Show: {show}\nTitle: {title}\nURL: {url}\n"
    "Time Range: {start_hms} - {end_hms}\n\n"
    "Provide a concise summary with:\n"
    "- Main topics discussed\n"
    "- Key points and takeaways\n"
    "- Notable quotes (if any)\n\n"
    'Transcript:\n"""{transcript}"""'
)


# ----------------------------- Utilities -----------------------------

def _get_random_proxy() -> Optional[str]:
    """Read proxy list file and return a random proxy."""
    import random
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

def fetch_podcast_metadata(url: str, use_cookies: bool = False, use_proxy: bool = False) -> Dict[str, Any]:
    """
    Use yt-dlp to extract episode/page info without downloading audio.
    Works on many host pages (Buzzsprout, Transistor, Spotify/Open, Apple page, etc.)
    Uses cookies if use_cookies=True and YTDLP_COOKIES (Netscape format) is set in env.
    Uses proxy if use_proxy=True and YTDLP_PROXY_FILE is set in env.
    """
    logger.info(f"Fetching podcast metadata from URL: {url}")
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "extract_flat": False,
        "extractor_retries": 3,
        "forceipv4": True,
        "js_runtimes": {"bun": {"path": "/root/.bun/bin/bun"}}
    }
    
    if use_cookies:
        cookies_file = os.getenv("YTDLP_COOKIES")
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts["cookiefile"] = cookies_file
            logger.info(f"Using cookies file: {cookies_file}")
    
    if use_proxy:
        proxy = _get_random_proxy()
        if proxy:
            ydl_opts["proxy"] = proxy
            logger.info(f"Using proxy: {proxy}")

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.error(f"Failed to fetch metadata from {url}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Failed to fetch podcast metadata: {str(e)}") from e

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

def _transcribe_in_chunks(audio_path: str, chunk_duration: int, tmpdir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Split audio into chunks and transcribe each separately to avoid OOM on very long files.
    Returns combined transcript text and segments.
    """
    import subprocess
    from faster_whisper import WhisperModel  # type: ignore
    
    logger.info(f"Starting chunked transcription with {chunk_duration}s chunks")
    
    model_name = os.getenv("PODCAST_ASR_MODEL", "base")
    device = os.getenv("PODCAST_ASR_DEVICE", "cpu")
    compute_type = os.getenv("PODCAST_ASR_COMPUTE", "int8")
    beam_size = int(os.getenv("PODCAST_ASR_BEAM_SIZE", "1"))
    
    logger.info(f"Loading Whisper model: {model_name}")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    all_segments: List[Dict[str, Any]] = []
    chunk_num = 0
    current_offset = 0.0
    
    while True:
        chunk_num += 1
        chunk_path = os.path.join(tmpdir, f"chunk_{chunk_num}.wav")
        
        logger.info(f"Extracting chunk {chunk_num} starting at {current_offset}s")
        
        # Extract chunk using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(current_offset),
            '-t', str(chunk_duration),
            '-ar', '16000',  # Whisper expects 16kHz
            '-ac', '1',  # Mono
            '-c:a', 'pcm_s16le',  # PCM encoding
            chunk_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg chunk extraction failed: {result.stderr}")
                break
            
            # Check if chunk was created and has content
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 1000:
                logger.info("No more audio to process")
                break
                
        except Exception as e:
            logger.error(f"Failed to extract chunk: {str(e)}")
            break
        
        # Transcribe chunk
        logger.info(f"Transcribing chunk {chunk_num}...")
        try:
            segments, _ = model.transcribe(
                chunk_path,
                vad_filter=True,
                beam_size=beam_size,
                best_of=1,
                patience=1.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,  # Each chunk is independent
            )
            
            chunk_segs = []
            for seg in segments:
                chunk_segs.append({
                    "text": seg.text.strip(),
                    "start": float(seg.start) + current_offset,  # Add offset for correct timestamp
                    "duration": float(seg.end - seg.start)
                })
            
            all_segments.extend(chunk_segs)
            logger.info(f"Chunk {chunk_num}: processed {len(chunk_segs)} segments")
            
        except Exception as e:
            logger.error(f"Failed to transcribe chunk {chunk_num}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Continue with next chunk instead of failing completely
        finally:
            # Clean up chunk file to save space
            try:
                os.remove(chunk_path)
            except:
                pass
        
        current_offset += chunk_duration
    
    logger.info(f"Chunked transcription complete. Total segments: {len(all_segments)}")
    text = " ".join(s["text"] for s in all_segments if s.get("text"))
    return text, all_segments if all_segments else None


def try_transcript_via_asr(url: str, use_cookies: bool = False, use_proxy: bool = False) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Download audio (yt-dlp) and transcribe locally using faster-whisper.
    Requires: faster-whisper + ffmpeg in your image, and PODCAST_ASR_ENABLE=1.
    Uses cookies if use_cookies=True and YTDLP_COOKIES (Netscape format) is set in env.
    Uses proxy if use_proxy=True and YTDLP_PROXY_FILE is set in env.
    """
    asr_enabled = os.getenv("PODCAST_ASR_ENABLE", "0")
    logger.info(f"ASR transcription attempt for URL: {url}")
    logger.info(f"PODCAST_ASR_ENABLE={asr_enabled}")
    
    if asr_enabled != "1":
        logger.warning("ASR is disabled (PODCAST_ASR_ENABLE != 1)")
        return None, None
    
    try:
        from faster_whisper import WhisperModel  # type: ignore
        logger.info("faster-whisper module imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import faster-whisper: {str(e)}")
        logger.error("Make sure faster-whisper is installed: pip install faster-whisper")
        raise RuntimeError(f"faster-whisper not installed or import failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error importing faster-whisper: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Unexpected error importing faster-whisper: {str(e)}") from e

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "extractor_retries": 3,
        "forceipv4": True,
        "format": "bestaudio/best",
        "outtmpl": "%(id)s.%(ext)s",
        "js_runtimes": {"bun": {"path": "/root/.bun/bin/bun"}}
    }
    
    if use_cookies:
        cookies_file = os.getenv("YTDLP_COOKIES")
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts["cookiefile"] = cookies_file
            logger.info(f"Using cookies file for download: {cookies_file}")
    
    if use_proxy:
        proxy = _get_random_proxy()
        if proxy:
            ydl_opts["proxy"] = proxy
            logger.info(f"Using proxy for download: {proxy}")

    tmpdir = tempfile.mkdtemp(prefix="pod_asr_")
    audio_path = None
    logger.info(f"Created temporary directory: {tmpdir}")
    
    try:
        logger.info("Starting audio download with yt-dlp...")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            logger.info(f"Audio download completed. Video ID: {video_id}")
            
            # choose the downloaded file
            candidates = glob.glob(os.path.join(os.getcwd(), f"{video_id}.*"))
            logger.info(f"Found {len(candidates)} audio file candidates: {candidates}")
            
            if candidates:
                audio_path = candidates[0]
                # move into tmpdir for cleanup
                new_path = os.path.join(tmpdir, os.path.basename(audio_path))
                shutil.move(audio_path, new_path)
                audio_path = new_path
                logger.info(f"Audio file moved to: {audio_path}")

        if not audio_path:
            logger.error("No audio file was downloaded")
            raise RuntimeError("Audio download failed: no file found after yt-dlp extraction")

        # Get audio duration to decide on chunking strategy
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            duration_seconds = float(result.stdout.strip()) if result.stdout.strip() else 0
            logger.info(f"Audio duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.1f} hours)")
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {str(e)}")
            duration_seconds = 0

        # For very long files (>2 hours), use chunked processing to avoid OOM
        max_chunk_duration = int(os.getenv("ASR_CHUNK_DURATION", "3600"))  # Default: 1 hour chunks
        if duration_seconds > 7200:  # > 2 hours
            logger.info(f"Long audio detected ({duration_seconds/3600:.1f}h). Using chunked transcription with {max_chunk_duration}s chunks")
            return _transcribe_in_chunks(audio_path, max_chunk_duration, tmpdir)

        model_name = os.getenv("PODCAST_ASR_MODEL", "base")
        device = os.getenv("PODCAST_ASR_DEVICE", "cpu")
        compute_type = os.getenv("PODCAST_ASR_COMPUTE", "int8")
        beam_size = int(os.getenv("PODCAST_ASR_BEAM_SIZE", "1"))  # Lower beam size = less memory
        
        logger.info(f"Initializing Whisper model with: model={model_name}, device={device}, compute_type={compute_type}, beam_size={beam_size}")
        logger.info("For long podcasts, consider: model=tiny, compute_type=int8, beam_size=1")
        
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load Whisper model '{model_name}' on device '{device}': {str(e)}") from e
        
        logger.info(f"Starting transcription of: {audio_path}")
        try:
            segments, info = model.transcribe(
                audio_path, 
                vad_filter=True,
                beam_size=beam_size,
                best_of=1,  # Don't sample multiple completions
                patience=1.0,  # Don't wait for better results
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )
            logger.info("Transcription started, processing segments...")
        except MemoryError as e:
            logger.error(f"Out of memory during transcription: {str(e)}")
            raise RuntimeError(
                f"Out of memory during transcription. Try: "
                f"1) Use smaller model: PODCAST_ASR_MODEL=tiny, "
                f"2) Ensure int8: PODCAST_ASR_COMPUTE=int8, "
                f"3) Reduce beam: PODCAST_ASR_BEAM_SIZE=1, "
                f"4) Increase Docker/container memory limits"
            ) from e
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Audio transcription failed during processing: {str(e)}") from e
        segs_list: List[Dict[str, Any]] = []
        try:
            for seg in segments:
                segs_list.append({
                    "text": seg.text.strip(),
                    "start": float(seg.start),
                    "duration": float(seg.end - seg.start)
                })
            logger.info(f"Processed {len(segs_list)} transcript segments")
        except Exception as e:
            logger.error(f"Failed to process transcript segments: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to process transcript segments: {str(e)}") from e
        
        text = " ".join(s["text"] for s in segs_list if s.get("text"))
        logger.info(f"Transcription completed successfully. Text length: {len(text)} characters")
        return text, segs_list if segs_list else None
    
    except Exception as e:
        logger.error(f"ASR transcription failed: {str(e)}")
        logger.debug(traceback.format_exc())
        # Re-raise with context
        raise
    finally:
        logger.info(f"Cleaning up temporary directory: {tmpdir}")
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


def summarize_with_ollama(base_url: str, model: str, title: str, show: str, url: str, transcript: str, segments: Optional[List[Dict[str, Any]]] = None, context_length: int=15000, per_segment: bool=False, segment_duration: int=1800, system_prompt: Optional[str] = None, summary_prompt: Optional[str] = None, segment_prompt: Optional[str] = None) -> str:
    logger.info(f"Starting Ollama summarization. Base URL: {base_url}, Model: {model}")
    logger.info(f"Transcript length: {len(transcript)} characters, Per-segment: {per_segment}")
    
    if not _check_ollama(base_url):
        logger.error(f"Ollama server check failed at {base_url}")
        raise RuntimeError(f"{base_url} does not look like an Ollama server (/api/tags not OK).")

    # Use provided prompts or defaults
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    sum_prompt = summary_prompt or DEFAULT_PODCAST_SUMMARY_PROMPT
    seg_prompt = segment_prompt or DEFAULT_PODCAST_SEGMENT_PROMPT

    def full_prompt(text: str) -> str:
        # Use summary prompt with system prompt prepended
        return (
            sys_prompt + "\n\n" +
            sum_prompt.format(show=show, title=title, url=url, transcript=text)
        )

    # Per-segment mode: independent summaries for each time segment
    if per_segment and segments:
        time_chunks = chunk_segments_by_duration(segments, duration_seconds=segment_duration)
        logger.info(f"Created {len(time_chunks)} time-based chunks for summarization")
        
        if not time_chunks:
            logger.warning("No time chunks created from segments")
            return "# Summary\n\n_(No content to summarize)_"
        
        result_parts: List[str] = []
        for i, (chunk_text, start_sec, end_sec) in enumerate(time_chunks, 1):
            start_hms = fmt_hms(int(start_sec))
            end_hms = fmt_hms(int(end_sec))
            logger.info(f"Summarizing segment {i}/{len(time_chunks)}: {start_hms} - {end_hms}")
            
            # Use segment prompt with system prompt prepended
            segment_prompt_text = (
                sys_prompt + "\n\n" +
                seg_prompt.format(
                    show=show, title=title, url=url, start_hms=start_hms, end_hms=end_hms, transcript=chunk_text
                )
            )
            
            summary = call_ollama_any(base_url, model, segment_prompt_text, context_length)
            result_parts.append(f"## Segment {i}: {start_hms} - {end_hms}\n\n{summary}")
        
        return "# Summary\n\n" + "\n\n".join(result_parts)

    # Default: full transcript summary
    return call_ollama_any(base_url, model, full_prompt(transcript), context_length)


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
    no_summary: bool = False,
    include_transcript: bool = False,
    context_length: int = 15000,
    per_segment: bool = False,
    segment_duration: int = 1800,
    use_cookies: bool = False,
    use_proxy: bool = False,
    system_prompt: Optional[str] = None,
    summary_prompt: Optional[str] = None,
    segment_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entrypoint (mirrors youtube_notes.process_youtube signature).
    
    Returns:
      { "meta": {...}, "summary": "<md>", "note_path": "/abs/path.md" }
    Raises on hard failures (so RQ can record them).
    
    context_length: Ollama context window size (num_ctx).
    per_segment: If True, create independent summaries for each time segment.
    segment_duration: Duration in seconds for each segment (default: 1800 = 30 minutes).
    use_cookies: If True, use YTDLP_COOKIES file for authenticated content.
    use_proxy: If True, use YTDLP_PROXY_FILE for proxy rotation.
    system_prompt: Optional system prompt (prepended to all prompts).
    summary_prompt: Optional full summary prompt template.
    segment_prompt: Optional segment prompt template.
    """
    load_dotenv()

    vault = vault or os.getenv("OBSIDIAN_VAULT")
    folder = folder if folder is not None else (os.getenv("OBSIDIAN_FOLDER") or "")
    langs = langs or [s.strip() for s in (os.getenv("PODCAST_LANGS") or os.getenv("YT_LANGS","")).split(",") if s.strip()] or DEFAULT_LANGS
    ollama_base = ollama_base or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # 1) Metadata
    meta = fetch_podcast_metadata(podcast_url, use_cookies=use_cookies, use_proxy=use_proxy)

    # 2) Transcript (ASR only - download and transcribe audio)
    logger.info(f"Starting podcast processing for: {podcast_url}")
    logger.info(f"Vault: {vault}, Folder: {folder}, No summary: {no_summary}")
    logger.info(f"Use cookies: {use_cookies}, Use proxy: {use_proxy}")
    
    try:
        transcript_text, transcript_segments = try_transcript_via_asr(podcast_url, use_cookies=use_cookies, use_proxy=use_proxy)
    except Exception as e:
        logger.error(f"Transcription failed for {podcast_url}: {str(e)}")
        raise  # Re-raise with original context

    # 3) Body
    if no_summary:
        logger.info("Skipping summary generation (no_summary=True)")
        body = make_metadata_only_body(meta, transcript_text if include_transcript else None)
    else:
        if not transcript_text:
            logger.error("No transcript text available for summarization")
            raise RuntimeError(
                "Audio transcription failed. No transcript text was generated. "
                "Check the logs above for specific errors. "
                "Ensure PODCAST_ASR_ENABLE=1 and faster-whisper is installed."
            )
        try:
            body = summarize_with_ollama(
                base_url=ollama_base,
                model=model,
                title=meta.get("episode_title") or "",
                show=meta.get("show") or "",
                url=meta.get("url") or podcast_url,
                transcript=transcript_text,
                segments=transcript_segments,
                context_length=context_length,
                per_segment=per_segment,
                segment_duration=segment_duration,
            )
            logger.info("Summarization completed successfully")
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    # 4) Write note
    note_path = None
    if vault:
        logger.info(f"Writing Obsidian note to vault: {vault}")
        try:
            note_path = str(write_obsidian_note(Path(vault).expanduser().resolve(), folder or "", meta, body))
            logger.info(f"Note written successfully to: {note_path}")
        except Exception as e:
            logger.error(f"Failed to write note: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    logger.info("Podcast processing completed successfully")
    return {"meta": meta, "summary": body, "note_path": note_path}
