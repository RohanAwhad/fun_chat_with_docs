import os
import redis

from fastapi import FastAPI
from pydantic import BaseModel

CRAWLER_URL = os.getenv("CRAWLER_URL", "http://localhost:5000")
READABLE_URL = os.getenv("READABLE_URL", "http://localhost:5001")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", 6379),
    db=os.getenv("REDIS_DB", 0),
)

app = FastAPI()
