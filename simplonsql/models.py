from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Date, Float

Base = declarative_base()

   
class LoanData(Base):
    __tablename__ = 'loandatas'
    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    taille = Column(Float)
    poids = Column(Float)
    nb_enfants = Column(Integer)
    quotient_caf = Column(Float)
    sport_licence = Column(String)
    niveau_etude = Column(String)
    region = Column(String)
    smoker = Column(String)
    revenu_estime_mois = Column(Integer)
    situation_familiale = Column(String)
    risque_personnel = Column(Float)
    loyer_mensuel = Column(Float)
    montant_pret = Column(Float)
