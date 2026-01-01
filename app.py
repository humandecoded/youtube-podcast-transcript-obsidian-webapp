import os, time
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from redis import Redis
from rq import Queue
from rq.job import Job
from werkzeug.utils import secure_filename
from services.youtube_notes import process_youtube, DEFAULT_LANGS
from services.podcast_notes import process_podcast, process_local_video

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Configure upload settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/video_uploads")
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v', 'mpg', 'mpeg', 'mp3', 'wav', 'm4a', 'ogg', 'aac'}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")) * 1024 * 1024  # Default 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Redis / RQ
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)
queue_name = os.getenv("RQ_QUEUE", "yt")
q = Queue(queue_name, connection=redis_conn, default_timeout=int(os.getenv("RQ_JOB_TIMEOUT", "3600")))

@app.get("/")
def index():
    # Import here to get the defaults after env is loaded
    from services.youtube_notes import DEFAULT_SYSTEM_PROMPT, DEFAULT_SUMMARY_PROMPT, DEFAULT_SEGMENT_PROMPT
    from services.podcast_notes import DEFAULT_PODCAST_SUMMARY_PROMPT, DEFAULT_PODCAST_SEGMENT_PROMPT
    
    return render_template("index.html", defaults={
        "vault": os.getenv("OBSIDIAN_VAULT", ""),
        "folder": os.getenv("OBSIDIAN_FOLDER", ""),
        "langs": os.getenv("YT_LANGS", ",".join(DEFAULT_LANGS)),
        "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "context_length": os.getenv("OLLAMA_CONTEXT_LENGTH", "4096"),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "youtube_summary_prompt": DEFAULT_SUMMARY_PROMPT,
        "youtube_segment_prompt": DEFAULT_SEGMENT_PROMPT,
        "podcast_summary_prompt": DEFAULT_PODCAST_SUMMARY_PROMPT,
        "podcast_segment_prompt": DEFAULT_PODCAST_SEGMENT_PROMPT,
    })

@app.post("/podcast")
def podcast():
    url = (request.form.get("podcast_url") or "").strip()
    
    # Get context length from form, default to 15000 if not provided or invalid
    try:
        context_length = int(request.form.get("context_length", 15000))
        if context_length < 512:
            context_length = 15000
    except Exception:
        context_length = 15000
    
    # Get segment duration from form, default to 1800 (30 minutes) if not provided or invalid
    try:
        segment_duration = int(request.form.get("segment_duration", 1800))
        if segment_duration < 60:  # minimum 1 minute
            segment_duration = 1800
    except Exception:
        segment_duration = 1800
    
    system_prompt = (request.form.get("system_prompt") or "").strip() or None
    summary_prompt = (request.form.get("summary_prompt") or "").strip() or None
    segment_prompt = (request.form.get("segment_prompt") or "").strip() or None
    
    job = q.enqueue(
        process_podcast,
        args=(url,),
        kwargs=dict(
            folder=(request.form.get("folder") or ""),
            langs=[s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()] or None,
            ollama_base=(request.form.get("ollama_base") or None),
            model=(request.form.get("model") or None),
            no_summary=request.form.get("no_summary") == "on",
            include_transcript=request.form.get("include_transcript") == "on",
            context_length=context_length,
            per_segment=request.form.get("per_segment") == "on",
            segment_duration=segment_duration,
            use_cookies=request.form.get("use_cookies") == "on",
            use_proxy=request.form.get("use_proxy") == "on",
            system_prompt=system_prompt,
            summary_prompt=summary_prompt,
            segment_prompt=segment_prompt,
        ),
        description=f"Podcast→Obsidian for {url}",
        result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
        failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
    )
    return redirect(url_for("job_status", job_id=job.get_id()))

@app.post("/upload")
def upload_video():
    # Check if file was uploaded
    if 'video_file' not in request.files:
        flash("No file selected", "error")
        return redirect(url_for("index"))
    
    file = request.files['video_file']
    
    if file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for("index"))
    
    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "error")
        return redirect(url_for("index"))
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
    except Exception as e:
        flash(f"Failed to save file: {str(e)}", "error")
        return redirect(url_for("index"))
    
    # Get context length from form
    try:
        context_length = int(request.form.get("context_length", 15000))
        if context_length < 512:
            context_length = 15000
    except Exception:
        context_length = 15000
    
    # Get segment duration from form
    try:
        segment_duration = int(request.form.get("segment_duration", 1800))
        if segment_duration < 60:
            segment_duration = 1800
    except Exception:
        segment_duration = 1800
    
    system_prompt = (request.form.get("system_prompt") or "").strip() or None
    summary_prompt = (request.form.get("summary_prompt") or "").strip() or None
    segment_prompt = (request.form.get("segment_prompt") or "").strip() or None
    
    # Enqueue processing job
    job = q.enqueue(
        process_local_video,
        args=(filepath,),
        kwargs=dict(
            filename=filename,
            folder=(request.form.get("folder") or ""),
            ollama_base=(request.form.get("ollama_base") or None),
            model=(request.form.get("model") or None),
            no_summary=request.form.get("no_summary") == "on",
            include_transcript=request.form.get("include_transcript") == "on",
            context_length=context_length,
            per_segment=request.form.get("per_segment") == "on",
            segment_duration=segment_duration,
            system_prompt=system_prompt,
            summary_prompt=summary_prompt,
            segment_prompt=segment_prompt,
        ),
        description=f"Local Video→Obsidian for {filename}",
        result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
        failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
    )
    return redirect(url_for("job_status", job_id=job.get_id()))

@app.post("/summarize")
def summarize():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        flash("Please paste a YouTube URL.", "error")
        return redirect(url_for("index"))

    # Gather options
    folder = (request.form.get("folder") or "").strip()
    langs = [s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()]
    ollama_base = (request.form.get("ollama_base") or "").strip() or None
    model = (request.form.get("model") or "").strip() or None
    no_summary = request.form.get("no_summary") == "on"
    include_transcript = request.form.get("include_transcript") == "on"
    use_cookies = request.form.get("use_cookies") == "on"
    use_proxy = request.form.get("use_proxy") == "on"
    per_hour = request.form.get("per_hour") == "on"
    system_prompt = (request.form.get("system_prompt") or "").strip() or None
    summary_prompt = (request.form.get("summary_prompt") or "").strip() or None
    segment_prompt = (request.form.get("segment_prompt") or "").strip() or None

    # Get context length from form, default to 4096 if not provided or invalid
    try:
        context_length = int(request.form.get("context_length", 4096))
        if context_length < 512:
            context_length = 4096
    except Exception:
        context_length = 4096

    # Get segment duration from form, default to 3600 (1 hour) if not provided or invalid
    try:
        segment_duration = int(request.form.get("segment_duration", 3600))
        if segment_duration < 60:  # minimum 1 minute
            segment_duration = 3600
    except Exception:
        segment_duration = 3600

    job = q.enqueue(
        process_youtube,
        args=(url,),
        kwargs={
            "folder": folder,
            "langs": langs or None,
            "ollama_base": ollama_base,
            "model": model,
            "no_summary": no_summary,
            "include_transcript": include_transcript,
            "use_proxy": use_proxy,
            "context_length": context_length,
            "per_hour": per_hour,
            "segment_duration": segment_duration,
            "per_hour": per_hour,
            "use_cookies": use_cookies,
            "system_prompt": system_prompt,
            "summary_prompt": summary_prompt,
            "segment_prompt": segment_prompt,
        },
        description=f"YouTube→Obsidian for {url}",
        url=url,
        result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
        failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
    )

    return redirect(url_for("job_status", job_id=job.get_id()))

@app.get("/job/<job_id>")
def job_status(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        flash("Unknown job ID.", "error")
        return redirect(url_for("index"))

    status = job.get_status(refresh=True)
    result = job.result if job.is_finished else None
    error = None
    
    if job.is_failed:
        error = str(job.exc_info).splitlines()[-1][:1000] if job.exc_info else "Job failed"

    # Get progress logs from job.meta
    logs = job.meta.get('logs', []) if hasattr(job, 'meta') and job.meta else []
    current_step = job.meta.get('current_step', '') if hasattr(job, 'meta') and job.meta else ''

    # Auto-refresh while pending/started
    refresh_secs = 2 if status in ("queued", "started", "deferred") else 0
    return render_template("result_async.html",
                           job_id=job_id, status=status, result=result, error=error,
                           refresh_secs=refresh_secs, logs=logs, current_step=current_step)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5050")))

