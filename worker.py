# worker.py
import os
from dotenv import load_dotenv
from redis import Redis
from rq import Worker, Queue

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "yt")

if __name__ == "__main__":
    conn = Redis.from_url(REDIS_URL)
    q = Queue(QUEUE_NAME, connection=conn)
    worker = Worker([q], connection=conn)
    worker.work()
