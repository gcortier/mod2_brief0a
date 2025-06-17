import sys
import pandas as pd
from sqlalchemy import create_engine, text
from simplonsql.models import LoanData, Base

## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru
logger = setup_loguru("logs/inject_data.log")


# Configuration de la base de données
SQLALCHEMY_DATABASE_URL = "postgresql://admin:changeme@localhost/simplon"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Charger les données du fichier CSV
def load_csv_to_db(csv_path):
    # Lire le fichier CSV
    df = pd.read_csv(csv_path)

    # Créer les tables si elles n'existent pas
    Base.metadata.create_all(bind=engine)

    # Insérer les données dans la base de données
    with engine.connect() as connection:
        df.to_sql(LoanData.__tablename__, con=connection, if_exists='append', index=False)
        logger.info(f"Données du fichier CSV '{csv_path}' insérées avec succès dans la base de données.")

def reset_database():
     # Supprimer toutes les tables
    Base.metadata.drop_all(bind=engine)
    logger.info("Toutes les tables ont été supprimées.")

    # Recréer toutes les tables
    Base.metadata.create_all(bind=engine)
    print("Toutes les tables ont été recréées.")

# Réinitialiser la table alembic_version
def reset_alembic_version():
    """
    Réinitialise la table alembic_version en supprimant toutes les entrées.
    """
    with engine.connect() as connection:
        connection.execute(text("DELETE FROM alembic_version;"))
        print("Table alembic_version réinitialisée avec succès.")



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset_alembic":
        # python inject_data.py reset_alembic
        reset_alembic_version()
    elif len(sys.argv) > 1 and sys.argv[1] == "init":
        # python inject_data.py init
        csv_path = "data/df_data_all_cleaned.csv"  # Chemin vers le fichier CSV
        load_csv_to_db(csv_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        # python inject_data.py reset
        reset_database()
