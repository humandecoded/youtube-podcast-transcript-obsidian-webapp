import os, time
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from redis import Redis
from rq import Queue
from rq.job import Job
from services.youtube_notes import process_youtube, DEFAULT_LANGS
from services.podcast_notes import process_podcast

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Redis / RQ
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)
queue_name = os.getenv("RQ_QUEUE", "yt")
q = Queue(queue_name, connection=redis_conn, default_timeout=int(os.getenv("RQ_JOB_TIMEOUT", "3600")))

# NTFY configuration
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
NTFY_AUTH_TOKEN = os.getenv("NTFY_AUTH_TOKEN", "")

def send_ntfy_notification(title, message, priority="default", tags=""):
    """Send notification via NTFY"""
    if not NTFY_TOPIC:
        return False
    
    try:
        headers = {
            "Title": title,
            "Priority": priority,
            "Tags": tags
        }
        
        if NTFY_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {NTFY_AUTH_TOKEN}"
        
        response = requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=message,
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"NTFY notification failed: {e}")
        return False

def create_job_with_notification(func, args, kwargs, description, notify=False, url=""):
    """Create a job and optionally set up notification"""
    job = q.enqueue(
        func,
        args=args,
        kwargs=kwargs,
        description=description,
        result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
        failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
    )
    
    # Store notification preference in job meta
    if notify and NTFY_TOPIC:
        job.meta["notify"] = True
        job.meta["url"] = url
        job.save_meta()
    
    return job

@app.get("/")
def index():
    return render_template("index.html", defaults={
        "vault": os.getenv("OBSIDIAN_VAULT", ""),
        "folder": os.getenv("OBSIDIAN_FOLDER", ""),
        "langs": os.getenv("YT_LANGS", ",".join(DEFAULT_LANGS)),
        "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "context_length": os.getenv("OLLAMA_CONTEXT_LENGTH", "4096"),
        "chunk_size": os.getenv("CHUNK_SIZE", "15000"),
        "ntfy_enabled": bool(NTFY_TOPIC),
    })

@app.post("/podcast")
def podcast():
    url = (request.form.get("podcast_url") or "").strip()
    notify = request.form.get("notify_ntfy") == "on"
    
    # Get chunk size from form, default to 15000 if not provided or invalid
    try:
        chunk_size = int(request.form.get("chunk_size", 15000))
        if chunk_size < 1000:
            chunk_size = 15000
    except Exception:
        chunk_size = 15000
    
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
    
    job = create_job_with_notification(
        process_podcast,
        args=(url,),
        kwargs=dict(
            vault=(request.form.get("vault") or None),
            folder=(request.form.get("folder") or ""),
            langs=[s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()] or None,
            ollama_base=(request.form.get("ollama_base") or None),
            model=(request.form.get("model") or None),
            map_reduce=request.form.get("map_reduce") == "on",
            no_summary=request.form.get("no_summary") == "on",
            include_transcript=request.form.get("include_transcript") == "on",
            chunk_size=chunk_size,
            context_length=context_length,
            per_segment=request.form.get("per_segment") == "on",
            segment_duration=segment_duration,
        ),
        description=f"Podcast→Obsidian for {url}",
        notify=notify,
        url=url
    )
    return redirect(url_for("job_status", job_id=job.get_id()))

@app.post("/summarize")
def summarize():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        flash("Please paste a YouTube URL.", "error")
        return redirect(url_for("index"))

    notify = request.form.get("notify_ntfy") == "on"
    print(notify)
    # Gather options
    vault = (request.form.get("vault") or "").strip() or None
    folder = (request.form.get("folder") or "").strip()
    langs = [s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()]
    ollama_base = (request.form.get("ollama_base") or "").strip() or None
    model = (request.form.get("model") or "").strip() or None
    map_reduce = request.form.get("map_reduce") == "on"
    no_summary = request.form.get("no_summary") == "on"
    include_transcript = request.form.get("include_transcript") == "on"
    per_hour = request.form.get("per_hour") == "on"

    # Get chunk size from form, default to 15000 if not provided or invalid
    try:
        chunk_size = int(request.form.get("chunk_size", 15000))
        if chunk_size < 1000:
            chunk_size = 15000
    except Exception:
        chunk_size = 15000

    # Get context length from form, default to 4096 if not provided or invalid
    try:
        context_length = int(request.form.get("context_length", 4096))
        if context_length < 512:
            context_length = 4096
    except Exception:
        context_length = 4096

    job = create_job_with_notification(
        process_youtube,
        args=(url,),
        kwargs={
            "vault": vault,
            "folder": folder,
            "langs": langs or None,
            "ollama_base": ollama_base,
            "model": model,
            "map_reduce": map_reduce,
            "no_summary": no_summary,
            "include_transcript": include_transcript,
            "chunk_size": chunk_size,
            "context_length": context_length,
            "per_hour": per_hour,
        },
        description=f"YouTube→Obsidian for {url}",
        notify=notify,
        url=url
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
    
    # Check if we need to send notification
    if job.is_finished or job.is_failed:
        if job.meta.get("notify") and not job.meta.get("notification_sent"):
            url = job.meta.get("url", "")
            if job.is_finished:
                title = "Processing Complete"
                message = f"Successfully processed: {url}"
                tags = "white_check_mark"
            else:
                title = "Processing Failed"
                message = f"Failed to process: {url}"
                tags = "x"
                
            if send_ntfy_notification(title, message, tags=tags):
                job.meta["notification_sent"] = True
                job.save_meta()
    
    if job.is_failed:
        error = str(job.exc_info).splitlines()[-1][:1000] if job.exc_info else "Job failed"

    # Auto-refresh while pending/started
    refresh_secs = 3 if status in ("queued", "started", "deferred") else 0
    return render_template("result_async.html",
                           job_id=job_id, status=status, result=result, error=error,
                           refresh_secs=refresh_secs)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5050")))

