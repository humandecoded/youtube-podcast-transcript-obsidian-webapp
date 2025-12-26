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

@app.get("/")
def index():
    return render_template("index.html", defaults={
        "vault": os.getenv("OBSIDIAN_VAULT", ""),
        "folder": os.getenv("OBSIDIAN_FOLDER", ""),
        "langs": os.getenv("YT_LANGS", ",".join(DEFAULT_LANGS)),
        "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "context_length": os.getenv("OLLAMA_CONTEXT_LENGTH", "4096"),
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
    
    job = q.enqueue(
        process_podcast,
        args=(url,),
        kwargs=dict(
            vault=(request.form.get("vault") or None),
            folder=(request.form.get("folder") or ""),
            langs=[s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()] or None,
            ollama_base=(request.form.get("ollama_base") or None),
            model=(request.form.get("model") or None),
            no_summary=request.form.get("no_summary") == "on",
            include_transcript=request.form.get("include_transcript") == "on",
            context_length=context_length,
            per_segment=request.form.get("per_segment") == "on",
            segment_duration=segment_duration,
        ),
        description=f"Podcast→Obsidian for {url}",
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
    vault = (request.form.get("vault") or "").strip() or None
    folder = (request.form.get("folder") or "").strip()
    langs = [s.strip() for s in (request.form.get("langs") or "").split(",") if s.strip()]
    ollama_base = (request.form.get("ollama_base") or "").strip() or None
    model = (request.form.get("model") or "").strip() or None
    no_summary = request.form.get("no_summary") == "on"
    include_transcript = request.form.get("include_transcript") == "on"
    per_hour = request.form.get("per_hour") == "on"

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
            "vault": vault,
            "folder": folder,
            "langs": langs or None,
            "ollama_base": ollama_base,
            "model": model,
            "no_summary": no_summary,
            "include_transcript": include_transcript,
            "context_length": context_length,
            "per_hour": per_hour,
            "segment_duration": segment_duration,
            "per_hour": per_hour,
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

    # Auto-refresh while pending/started
    refresh_secs = 3 if status in ("queued", "started", "deferred") else 0
    return render_template("result_async.html",
                           job_id=job_id, status=status, result=result, error=error,
                           refresh_secs=refresh_secs)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5050")))

