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
├── logs/
│   ├── alchemy-api.log
│   └── main_api.log
├── modules/
│   ├── evaluate.py
│   ├── preprocess.py
│   └── print_draw.py
├── .gitignore
├── README.md
├── main_api.py => API FastAPI 
├── alchemy_api.py => Niveau 0 API
└── requirements.txt
```

###### `data/`
* `data-all-complete-684bf9cd92797851623245.csv` : les données du dernier cas avec des colonnes en plus

###### `figures/`
Sauvegarde des images des courbes de coût et autres graphiques pour visualiser les performances de notre modèle.

###### `models/`
* `models.py` : La définition du model
* `model_20250620_1536_0_None.pkl` :Dernier model entrainé
* `preprocessor_latest.pkl` : Le preprocess avec les nouvelles colonnnes !

###### `modules/`
Ce sont nos couteaux suisses du code. Chaque fichier est un expert dans son domaine.
* `evaluate.py` : Le juge impitoyable qui dit si notre modèle est un génie ou un cancre.
* `preprocess.py` : Le chef cuisinier des données. Il les nettoie, les coupe, les assaisonne pour qu'elles soient parfaites pour notre IA.
* `print_draw.py` : L'artiste du groupe. Il transforme nos chiffres barbares en beaux graphiques pour que même ta grand-mère puisse comprendre (enfin, presque).

###### `notebooks/`
les différents notebook pour explorer les briefs


###### `simplonsql/`
Le model de données utilisé pour la base de données SQLAlchemy et les routes FastApi

---

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
- ### Installations des requis Mod3 Brief 2: 
```bash
pip install tensorflow
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
uvicorn main_api:app --host 127.0.0.1 --port 8000 --reload
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



- FastAPI : http://localhost:8000/docs
- MLflow UI : http://localhost:5000


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
- J'ai créé une DB PostgreSQL containerisé que j'ai rempli avec les valeurs  j'ai créé un CRUD sur la ressource Client. J'ai également créé un préprocesseur et j'ai créé un modèle que j'ai entraîné avec les données préprocessées de la DB.
- J'ai changé le preprocesseur pour s'adapter aux nouvelles colonnes du jeu de données.
- J'ai créé des scripts utilitaires pour injecter les données `python inject_data.py init`
- Ajout du module pydantic-sqlalchemy pour gérer les changements de modèles et rendre + dynamique pydantic
- Une route `/predict_loandata42` pour faire des prédictions sur des données envoyées via pydantic
- le loging des performances avec `loguru` et un setup simplifié pour le logger.
- les images des couts stockés dans le dossier `figures/`.


## Quelles difficultés j’ai rencontrées ? 
- l'apprentissage des nouvelles librairies et leur implémentation dans une architecture relativement propre, la réutilisation d"outils déja vu (MLFlow ...). 
- Je suis repassé sur sqllite en db car trop de soucis avec postgres
- La gestion des migrations de la base de données avec Alembic.
- La gestion des données manquantes et la création d'un préprocesseur efficace.


## Qu’est-ce que j’ai appris ? 
- J'ai appris à utiliser l'ORM SQLAlchemy et à gérer une DB avec.
- J'ai consolidé les outils à ma disposition pour créer une API REST et un modèle de machine learning : 
- Transférer des poids d'un modèle à un autre.
- Comment utiliser des stratégy de remplissage de colonne en gardans l'information originale : missing True/False
  - Simplification de l'initialisation 
  ```python 
    ## Base initialisation for Loguru and FastAPI
    from myapp_base import setup_loguru, app, Request, HTTPException
    logger = setup_loguru("logs/alchemy_api.log")
  ```





> **Note :** OK mes qustionnements on été répondu lors de la session en one to one

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
- Les risques de biais restants sont identifiés 

---


# Brief 2 :
Objectif : Adapter un modèle IA existant (TensorFlow/Keras) à de nouvelles colonnes d’entrée, sans perdre les poids internes, puis le réentraîner sur un dataset enrichi.
Suivi des performances : MLflow ou TensorBoard.
Exposition du modèle : API FastAPI prenant en compte les nouvelles colonnes.
Livrables attendus :
Nouveau modèle entraîné (.h5, .pkl ou SavedModel)
Script de modification de l’architecture + script d’entraînement
Script/projet FastAPI pour l’API
Journal de suivi MLflow
README détaillé


- j'ai refacto le code pour rendre dynamique les usages de colonnes pour l'entrainnement du model
- J'ai ajouté la stratégie de missing pour les colonnes incompletes (Remplissage + indication dans une colonne dédié pour savoir si la valeur etait manquante ou pas à la base)
- entraînement du model : python alchemy_api.py train
  
## Performance 1er entrainnement :
==================Performance for run 0/1===================
MSE: 77444596.8351, MAE: 6463.7837, R²: 0.3169
============================================================


- test avec notebook
- Codage Codage Codage...
- Ajout stratégie de transfert de poids en option de settings


## Exécution du script d'entraînement en transferant les poids du premier model
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
==================Performance for run 0/1===================
MSE: 61944948.3360, MAE: 5717.0758, R²: 0.4537
============================================================
Model enregistré dans models/model_20250620_1536_0_None.pkl
Courbe de loss : model_20250620_1536_0_None.jpg



- Création d'un script pour enregistrer le preprocess pour la route de prédiction
- Changement de type pour la données 

- Utilisation d'une data d'enregistrement de l'ordre des colonnes pour completer la route de prédiction.