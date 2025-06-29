from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Optional
## Base initialisation for Loguru and FastAPI
# from myapp_base import setup_loguru, app, Request, HTTPException
from alchemy_api import setup_loguru, app, Request, HTTPException
logger = setup_loguru("logs/main_api.log")

## SQLAlchemy imports
from fastapi import Depends
from sqlalchemy.orm import Session

from simplonsql.models import Base, LoanData
from typing import List

from pydantic_sqlalchemy import sqlalchemy_to_pydantic


# SQLALCHEMY_DATABASE_URL = "postgresql://admin:changeme@localhost/simplon"
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Création des tables si elles n'existent pas
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Modèle Pydantic dynamic ! pour LoanData
LoanDataSchema = sqlalchemy_to_pydantic(LoanData)

# Route GET pour récupérer toutes les entrées de LoanData
@app.get("/loandata", response_model=List[LoanDataSchema])
def get_loandata(request: Request, id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Récupère les données de LoanData.
    Si un id est fourni, retourne l'entrée correspondante.
    Sinon, retourne les 15 premières entrées.
    http://127.0.0.1:8000/loandata?id=42
    """
    
    logger.info(f"Route '{request.url.path}' avec id={id}")
    try:
        if id is not None:
            # Recherche par ID
            entry = db.query(LoanData).filter(LoanData.id == id).first()
            if not entry:
                raise HTTPException(status_code=404, detail=f"LoanData avec id={id} introuvable")
            return [entry]  # Retourne une liste contenant une seule entrée
        else:
            # Retourne les 15 premières entrées
            entries = db.query(LoanData).limit(15).all()
            return entries
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données LoanData: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Route POST pour ajouter une nouvelle entrée dans LoanData
@app.post("/loandata", response_model=LoanDataSchema)
def create_loandata(request: Request, loandata: LoanDataSchema, db: Session = Depends(get_db)):
    logger.info(f"Route '{request.url.path}'")
    try:
        new_entry = LoanData(**loandata.dict())
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        logger.info(f"Nouvelle entrée LoanData ajoutée avec l'id: {new_entry.id}")
        return new_entry
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout d'une entrée LoanData: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route PUT pour mettre à jour une entrée LoanData par id
@app.put("/loandata/{id}", response_model=LoanDataSchema)
def update_loandata(request: Request, id: int, loandata: LoanDataSchema, db: Session = Depends(get_db)):
    logger.info(f"Route '{request.url.path}' (update) pour id={id}")
    try:
        entry = db.query(LoanData).filter(LoanData.id == id).first()
        if not entry:
            logger.warning(f"Aucune entrée LoanData trouvée avec l'id: {id}")
            raise HTTPException(status_code=404, detail="Entrée LoanData non trouvée")
        # Mise à jour des champs
        for key, value in loandata.dict().items():
            setattr(entry, key, value)
        db.commit()
        db.refresh(entry)
        logger.info(f"Entrée LoanData avec l'id {id} mise à jour")
        return entry
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour d'une entrée LoanData: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Route DELETE pour supprimer une entrée LoanData par id
@app.delete("/loandata/{id}", response_model=dict)
def delete_loandata(request: Request, id: int, db: Session = Depends(get_db)):
    logger.info(f"Route '{request.url.path}'")
    try:
        loandata_entry = db.query(LoanData).filter(LoanData.id == id).first()
        if not loandata_entry:
            logger.warning(f"Aucune entrée LoanData trouvée avec l'id: {id}")
            raise HTTPException(status_code=404, detail="Entrée LoanData non trouvée")
        db.delete(loandata_entry)
        db.commit()
        logger.info(f"Entrée LoanData avec l'id {id} supprimée")
        return {"message": "Entrée supprimée avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors de la suppression d'une entrée LoanData: {e}")
        raise HTTPException(status_code=500, detail=str(e))
