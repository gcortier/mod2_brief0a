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

###### `data/` (Le garde-manger du projet)
Ici, c'est là que nos précieuses données vivent.
* `df_new.csv` : Les données fraîches du jour, prêtes à être dévorées par notre IA.
* `df_old.csv` : Les classiques, les vétérans, ceux qui ont tout vu. On les garde par nostalgie (et pour la rétrospective).

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

## Etape clen du jeu de données
- Utilisation du notebook créé lors du module 2 pour créer un jeu de données propre.
- Le jeu de données est stocké dans le dossier `data/` sous le nom `data-all-684bf775c031b265646213.csv`.
- Le notebook est disponible dans le dossier `notebooks/` sous le nom `ethique_data_cleaning.ipynb`.
