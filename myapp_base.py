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
    """
    Endpoint racine de l'API.
    Retourne un message de bienvenue : "Bienvenue sur l'API".
    Cette route est utilisée pour vérifier que l'API fonctionne correctement.
    """
    try:
        logger.info(f"Route '{request.url.path}' called by {request.client.host}")
        return {"response": "Bienvenue sur l'API"}
    except Exception as e:
        logger.error(f"Error logging request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    

        
@app.get("/health")
async def health(request: Request):
    """
    Endpoint de santé pour vérifier que l'application fonctionne.
    """
    logger.info(f"Route '{request.url.path}' called by {request.client.host}")
    return {"status": "healthy", "message": "API is running"}
