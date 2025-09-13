#!/usr/bin/env python3

import time
import os
import json
import logging
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables (define before routes)
global cv_detector_alive
cv_detector_alive = False

global doomscroll_stats
doomscroll_stats = {
    "doom_secs_today": 0,
    "last_doomscrolled_at": None,
    "doomscroll_clean_streak_secs": 0,
    "penalty_rate": 0.10,
    "is_doomscrolling": False, # simple toggle
}

app = FastAPI(title="Doomscrolling Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

@app.get("/test")
def test():
    print("TEST ENDPOINT HIT!", flush=True)
    logger.info("Test endpoint hit!")
    return {"message": "Hello, World!"}

# your API under /api
api = APIRouter(prefix="/api")

@api.get("/stats")
async def api_stats():
    logger.info("API stats requested")
    print("API STATS HIT!", flush=True)
    return doomscroll_stats

app.include_router(api)

# very simple endpoint to track accumulating doomscroll data
# simply:
@app.post("/data")
async def detection(request: Request):
    data = await request.json()
    logger.info(f"Received data: {data}")
    print(f"DEBUG: Received data: {data}", flush=True)

    if data["type"] == "doomscroll":
        cv_detector_alive = True
        doomscroll_stats["is_doomscrolling"] = True
        logger.info("Doomscrolling detected!")

        if doomscroll_stats["last_doomscrolled_at"] is None:
            doomscroll_stats["last_doomscrolled_at"] = time.time()

        doomscroll_stats["doom_secs_today"] += data["duration"]
        doomscroll_stats["doomscroll_clean_streak_secs"] = 0
        doomscroll_stats["owed_usd"] += data["duration"] * doomscroll_stats["penalty_rate"]

    elif data["type"] == "heartbeat":
        cv_detector_alive = True
        doomscroll_stats["is_doomscrolling"] = False
        logger.info("Heartbeat received - not doomscrolling")

        doomscroll_stats["doomscroll_clean_streak_secs"] = time.time() - doomscroll_stats["last_doomscrolled_at"] if doomscroll_stats["last_doomscrolled_at"] is not None else 0

    doomscroll_stats["owed_usd"] = (
        doomscroll_stats["doom_secs_today"] * doomscroll_stats["penalty_rate"]
    )

    logger.info(f"Updated stats: {doomscroll_stats}")
    return {"ok": True}

# serve static site at / (put this LAST so API routes work first)
# html=True lets index.html be served for '/'
app.mount("/", StaticFiles(directory="../web", html=True), name="web")

# uvicorn is a webserver, sorta like node. (asynchronous server gateway node, asgn)
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on 0.0.0.0:8000")
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True, log_level="debug", access_log=True)