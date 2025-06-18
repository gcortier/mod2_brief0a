from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_performance(y_true, y_pred):
    """
    Fonction pour mesurer les performances du modèle avec MSE, MAE et R².
    Basé sur le model MSE : quadratique de l'erreur moyenne,
    MAE : erreur absolue moyenne, 
    R² : coefficient de détermination. : proportion de la variance des données expliquée par le modèle. target : 1
    Args:
        y_true (array-like): Valeurs réelles.
        y_pred (array-like): Valeurs prédites par le modèle.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2} 