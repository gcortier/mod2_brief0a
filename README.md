# Mod 3 Exposer une base de données relationnelle via une API REST et entrainement d'un modèle
###### Le `.venv` 


```bash
python -m venv .venv
```


* **Windows (PowerShell) :**
    ```bash
    .\.venv\Scripts\Activate.ps1
    ```
* **Windows (CMD) :**
    ```bash
    .\.venv\Scripts\activate.bat
    ```
* **macOS / Linux :**
    ```bash
    source .venv/bin/activate
    ```


###### Le `requirements.txt`


Assure-toi que ton `.venv` est activé, puis :

```bash
pip install -r requirements.txt
```


#### 🗺️ Architecture : Où va quoi dans notre petit monde ? 🗺️


.
├── data/
│   ├── data-all-684bf775c031b265646213.csv
├── models/
│   ├── models.py
│   ├── model_2024_08.pkl
│   └── preprocessor.pkl
├── figures/
│   ├── ...
├── modules/
│   ├── evaluate.py
│   ├── preprocess.py
│   └── print_draw.py
├── .gitignore
├── README.md
├── app.py  => root de front
├── mlFlow_api.py => API FastAPI 
└── requirements.txt
```

###### `data/`
* `data-all-684bf775c031b265646213.csv` : Les données de base à traiter.

###### `figures/` (L'analyste visuelle)
Sauvegarde des images des courbes de coût et autres graphiques pour visualiser les performances de notre modèle.

###### `models/` (Le garage à cerveaux)
Ce dossier, c'est notre caverne d'Ali Baba des cerveaux artificiels.
* `models.py` : Les plans de nos futurs cyborgs... euh, de nos modèles. C'est ici que l'on définit l'architecture de nos NN et autres merveilles.
* `model_2024_08.pkl` : Une version sauvegardée de notre modèle. On l'a encapsulé pour qu'il ne s'échappe pas et ne domine pas le monde... pas encore.
* `preprocessor.pkl` : L'outil magique qui prépare les données avant de les donner à manger au modèle. Sans lui, c'est l'indigestion assurée !

###### `modules/` (La boîte à outils de MacGyver)
Ce sont nos couteaux suisses du code. Chaque fichier est un expert dans son domaine.
* `evaluate.py` : Le juge impitoyable qui dit si notre modèle est un génie ou un cancre.
* `preprocess.py` : Le chef cuisinier des données. Il les nettoie, les coupe, les assaisonne pour qu'elles soient parfaites pour notre IA.
* `print_draw.py` : L'artiste du groupe. Il transforme nos chiffres barbares en beaux graphiques pour que même ta grand-mère puisse comprendre (enfin, presque).

---

On espère que cette petite virée dans notre projet t'a plu. N'hésite pas à jeter un œil au `main.py` pour lancer le grand spectacle !

*Fait avec amour, code et une bonne dose de caféine (et un peu de folie).*


# TD => GOGOGO
## setup

- ### Génération requirements.txt à chaque installation de module
```bash
pip freeze > requirements.txt
```

- ### Installations des requis sqlAlchemy: 
```bash
pip install psycopg2-binary
pip install pydantic-sqlalchemy
```

- ### Installations des requis loguru: 
```bash
pip install loguru
```

- ### Installations des requis FastAPI/Streamlit: 
```bash
pip install nltk fastapi streamlit uvicorn requests pydantic
```
- #### Pour lancer le serveur MLflow :
```bash
uvicorn mlFlow_api:app --host 127.0.0.1 --port 8000 --reload
```
- #### Description des routes de l'API FastAPI :
[GET /docs](http://127.0.0.1:8000/docs#/)


- ### Installation des bibliothèques pour les tests unitaires: 
```bash
pip install pytest httpx
pytest test_predict_api.py
```

- ### Installations des requis pour MLflow : 
  > **mlFlow**
  MlFlow est un outil de gestion des expériences de machine learning. Il permet de suivre les expériences, de gérer les modèles et de visualiser les résultats.
```bash
pip install mlflow scikit-learn pandas matplotlib
```

# Pour lancer le serveur MLflow :
```bash
mlflow ui
```

## Streamlit
lancer le serveur Streamlit pour l'interface utilisateur :
```bash
streamlit run streamlit_app.py
```
### Pour accéder au front (root + pages entrainnement et prediction :)
[Streamlit Front](http://localhost:8501)


## Déroulé du travail
Création d'un script pour générer 3 entrainements et les stocker sur MLflow : 
Les models créé sont stockés dans le dossier `models/` et pictures du drawloss sont stockés dans le dossier `figures/`.


- J'ai mis en place :
  - Un script pour lancer 5 entrainements sur les anciennes données et 5 sur les nouvelles données.
  - le suivi des performances de chaque entrainement dans MLflow. (possibilité de stocker ou pas les modèles et les graph de loss)
  - les tests unitaires avec pytest
  - le loging des performances avec `loguru` et un setup simplifié pour le logger.
  - les images des couts stockés dans le dossier `figures/`.
  - Une route `/predict` pour faire des prédictions sur des données envoyées via questionnaire *Streamlit*.
  - Une route `/retrain` pour réentrainer le modèle (in progress).
  - Containerisation avec docker File puis docker-compose 
  - Stockage de l'id du model en local dans un fichier pour le retrouver depuis docker



**Docker : Build & Run**

Pour builder l’image Docker manuellement :
```bash
docker build -t mlflow-app .
```
Puis lancer le conteneur (tous services dans le même conteneur) :
```bash
docker run -p 8000:8000 -p 8501:8501 -p 5000:5000 mlflow-app
```

- FastAPI : http://localhost:8000/docs
- Streamlit : http://localhost:8501
- MLflow UI : http://localhost:5000

**Avec Docker Compose (meilleur approche)**

Pour builder et lancer tous les services (API, Streamlit, MLflow) dans des conteneurs séparés :
```bash
docker compose up --build
```

Pour tout arrêter proprement :
```bash
docker compose down
```


```powershell
wsl --unregister docker-desktop
wsl --unregister docker-desktop-data
```

## setup docker postgresql
```powershell
docker compose -f docker-compose-postgres.yml up -d
```

## Etape clés du projet
- J'ai créé un nouveau projet à partir des modules précédent : FastAPI / MLFlow / Streamlit/ LOGURU / Docker / 
- Le jeu de données est stocké dans le dossier `data/` sous le nom `data-all-684bf775c031b265646213.csv`.
- J'ai suivi le module sur la prise en main de SQL Alchimy et la gestion ORM
- Le notebook est disponible dans le dossier `notebooks/` sous le nom `SQLAlchemy_exploration.ipynb`.
- Utilisation du notebook créé lors du module 2 pour analyser et créer un jeu de données propre (CSV).
- Réutilisation du notebook créé lors du module 2 pour analyser le jeu de données, comprendre les données et les nettoyer.
- Le notebook est disponible dans le dossier `notebooks/` sous le nom `ethique_data_cleaning.ipynb`.


- J'ai créé un script `python alchemy_api.py clean_dataset` pour nettoyer le jeu de données et le stocker dans le dossier `data/`.
- 
- J'ai créé une DB PostgreSQL containerisé que j'ai rempli avec les valeurs  j'ai créé un CRUD sur la ressource Client. J'ai également créé un préprocesseur et j'ai créé un modèle que j'ai entraîné avec les données préprocessées de la DB.
- J'ai changé le preprocesseur pour s'adapter aux nouvelles colonnes du jeu de données.
Quelles difficultés j’ai rencontrées dans la journée ? 
- l'apprentissage des nouvelles librairies et leur implémentation dans une architecture relativement propre, la réutilisation d"outils déja vu (MLFlow ...). 
Qu’est-ce que j’ai appris ? 
- J'ai appris à utiliser l'ORM SQLAlchemy et à gérer une DB avec.
- J'ai consolidé les outils à ma disposition pour créer une API REST et un modèle de machine learning : 
  - Simplification de l'initialisation 
  ```python 
    ## Base initialisation for Loguru and FastAPI
    from myapp_base import setup_loguru, app, Request, HTTPException
    logger = setup_loguru("logs/alchemy_api.log")
  ```

- Je suis repassé sur sqllite en db car trop de soucis avec postgres

- J'ai créé des scripts utilitaires pour injecter les données `python inject_data.py init`

- Ajout du module pydantic-sqlalchemy pour gérer les changements de modèles et rendre + dynamique pydantic

> **Note :** A quoi sert le modèle ?

## Documentation du flux de données et évaluation éthique

### Flux de données et transformations

- Les données brutes sont importées depuis un fichier CSV (`data-all-complete-684bf9cd92797851623245.csv`).
- Un nettoyage éthique est réalisé dans le notebook `ethique_data_cleaning_complete.ipynb` :  
  - Suppression des colonnes sensibles ou non conformes (nom, prénom, sexe, nationalité, orientation sexuelle, date de création de compte).
  - Remplissage des valeurs manquantes pour certaines colonnes (`loyer_mensuel` par la moyenne, `situation_familiale` par la modalité la plus fréquente).
  - Filtrage des valeurs aberrantes (poids, nb_enfants, quotient_caf, loyer_mensuel).
  - Suppression des colonnes avec trop de valeurs manquantes (`score_credit`, `historique_credits`).
- Le dataset nettoyé est sauvegardé sous `df_data_all_complete_cleaned.csv` puis injecté dans la base via le script `inject_data.py`.

### Schéma de la base de données

- Table principale : `loandatas`
- Colonnes :  
  `id`, `age`, `taille`, `poids`, `nb_enfants`, `quotient_caf`, `sport_licence`, `niveau_etude`, `region`, `smoker`, `revenu_estime_mois`, `situation_familiale`, `risque_personnel`, `loyer_mensuel`, `montant_pret`
- Le schéma est versionné et documenté via les migrations Alembic.

### Évaluation éthique

- Les colonnes à risque de discrimination ou d’identification ont été supprimées.
- Les choix de remplissage et de filtrage sont documentés dans le notebook.
- Les risques de biais restants (ex : sur la région, le niveau d’étude, ou le remplissage par la moyenne) sont identifiés et à surveiller lors de l’utilisation des données.

---




