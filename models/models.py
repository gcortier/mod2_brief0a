from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_nn_model(input_dim):
    """
    Fonction pour créer et compiler un modèle de réseau de neurones simple.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0 ):
    """
    Entraine le modèle de réseau de neurones sur les données fournies.
    Args:
        model: Modèle de réseau de neurones à entraîner.
        X: Données d'entrée pour l'entraînement.
        y: Cibles pour l'entraînement.
        X_val: Données d'entrée pour la validation (optionnel).
        y_val: Cibles pour la validation (optionnel).
        epochs: Nombre d'époques pour l'entraînement (par défaut 50).
        batch_size: Taille du lot pour l'entraînement (par défaut 32).
        verbose: Niveau de verbosité de l'entraînement (par défaut 0, aucune sortie).
    """
    hist = model.fit(X, y, 
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model , hist

def model_predict(model, X):
    """
    Fonction pour prédire les valeurs cibles à partir des données d'entrée en utilisant le modèle de réseau de neurones.
    Args:
        model: Modèle de réseau de neurones entraîné : model_2024_08.pkl
        X: Données d'entrée pour la prédiction.
        Returns:
        y_pred: Prédictions du modèle sur les données d'entrée.
    """
        
    y_pred = model.predict(X).flatten()
    return y_pred