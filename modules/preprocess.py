from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# sert à séparer les données en ensembles d'entraînement et de test : 20% test, 80% train
def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocessing_v2(df, settings):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    
    numerical_cols = settings.get("numerical_cols", ["age", "taille", "poids", "revenu_estime_mois", "risque_personnel", "loyer_mensuel", "score_credit", "historique_credits"])    
    categorical_cols = settings.get("categorical_cols", ["sport_licence", "niveau_etude", "region", "smoker", "situation_familiale"])
    partials_num_cols = settings.get("partials_num_cols", [])
    # on ajoute les colonnes missing + la colonne _missing aux colonnes numériques
    numerical_cols += [col for col in partials_num_cols]
    numerical_cols += [col + "_missing" for col in partials_num_cols]
    
    
    
    # Vérification des colonnes manquantes
    missing_cols = [col for col in numerical_cols + categorical_cols if col not in df.columns]
    if missing_cols:
        print(f"Colonnes manquantes dans le DataFrame : {missing_cols}")
        

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    # X = df.drop(columns=["nom", "prenom", "montant_pret"], inplace=True, errors='ignore')
    X = df.drop(columns=["montant_pret"])
    y = df["montant_pret"]

    X_processed = preprocessor.fit_transform(X)
    print(f"Shape of X_processed: {X_processed.shape}")
    
    # sortie des colonnes utilisés
    # feature_names = preprocessor.get_feature_names_out()
    # print(len(feature_names))  # Affiche le nombre de colonnes générées
    # print(feature_names)       # Affiche les noms des colonnes générées

    return X_processed, y, preprocessor


    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    
    numerical_cols = ["age", "taille", "poids", "revenu_estime_mois", "risque_personnel", "loyer_mensuel", "score_credit", "historique_credits"]    
    categorical_cols = ["sport_licence", "niveau_etude", "region", "smoker", "situation_familiale"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    # X = df.drop(columns=["nom", "prenom", "montant_pret"], inplace=True, errors='ignore')
    X = df.drop(columns=["montant_pret"])
    y = df["montant_pret"]

    X_processed = preprocessor.fit_transform(X)
    print(f"Shape of X_processed: {X_processed.shape}")
    
    feature_names = preprocessor.get_feature_names_out()
    print(len(feature_names))  # Affiche le nombre de colonnes générées
    print(feature_names)       # Affiche les noms des colonnes générées
    
    return X_processed, y, preprocessor