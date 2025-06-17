import os
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

def setup_loguru(logfile="logs/app.log"):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger.remove()
    logger.add(
        logfile,
        rotation="500MB",
        retention="7 days",
        level="INFO",
        format="{time} {level} {message}"
    )
    return logger

def create_app():
    return FastAPI()


app = create_app()

class Response(BaseModel):
    response: str

@app.get("/")
async def root(request: Request):
    logger.info(f"Route '{request.url.path}' called by  {request.client.host}")
    return {"response": "Bienvenue sur l'API"}
