import matplotlib.pyplot as plt


def print_data(dico, exp_name="exp 1"):
    """
    Affiche les métriques de performance du modèle (MSE, MAE, R²) dans la console.
    
    Args:
        dico (dict): Dictionnaire contenant les clés 'MSE', 'MAE', 'R²'.
        exp_name (str, optional): Nom de l'expérience à afficher dans le titre. Par défaut "exp 1".
    """
    mse = dico["MSE"]
    mae = dico["MAE"]
    r2 = dico["R²"]
    print(f'{exp_name:=^60}')
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("="*60)


def draw_loss(history, save_path=None):
    """
    Affiche ou enregistre les courbes de loss et val_loss de l'historique d'entraînement d'un modèle.
    loss  = perte sur l'ensemble d'entraînement
    val_loss = perte sur l'ensemble de validation
    
    Args:
        history: Objet d'historique retourné par l'entraînement du modèle (contenant 'loss' et 'val_loss').
        save_path: Chemin du fichier pour enregistrer la figure. Si None, affiche la figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title('Courbes de Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()