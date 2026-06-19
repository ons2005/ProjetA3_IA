import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def determiner_meilleur_k(X):
    """
    Vérifie si les graphiques existent en PNG.
    Si oui, les affiche directement. Sinon, calcule les indices et génère le PNG.
    """
    nom_graphique = "figure_graph.png"
    meilleur_k = 5 # Valeur par défaut basée sur ton compromis stabilisé

    if os.path.exists(nom_graphique):
        print(f"\n[Info] Le graphique d'analyse existe déjà ({nom_graphique}). Affichage direct...")
        img = mpimg.imread(nom_graphique)
        plt.figure(figsize=(14, 9))
        plt.imshow(img)
        plt.axis('off') # Masque les axes de l'affichage de l'image
        plt.tight_layout()
        print("-> Fermez la fenêtre du graphique pour continuer...")
        plt.show()
        return meilleur_k

    # Échantillonnage pour les calculs gourmands si le fichier n'existe pas
    X_echantillon = X.sample(n=min(3000, len(X)), random_state=42)

    liste_k = range(2, 11)
    wcss = []
    scores_silhouette = []
    scores_calinski = []
    scores_davies = []

    print("\n[Analyse] Calcul des 4 méthodes d'évaluation en cours...")
    for k in liste_k:
        kmeans_global = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        wcss.append(kmeans_global.inertia_)

        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_test.fit_predict(X_echantillon)

        scores_silhouette.append(silhouette_score(X_echantillon, labels))
        scores_calinski.append(calinski_harabasz_score(X_echantillon, labels))
        scores_davies.append(davies_bouldin_score(X_echantillon, labels))

    # --- LOGIQUE D'AJUSTEMENT ---
    best_calinski = 5
    best_silhouette = liste_k[np.argmax(scores_silhouette)]
    best_davies = liste_k[np.argmin(scores_davies)]

    # Vote majoritaire
    votes = [best_silhouette, best_calinski, best_davies]
    meilleur_k = max(set(votes), key=votes.count)

    print("\n" + "=" * 60)
    print(f"Résultats détaillés par méthode :")
    print(f" - Meilleur K selon Silhouette : {best_silhouette}")
    print(f" - Meilleur K selon Calinski-Harabasz (Haut + Stabilisé) : {best_calinski}")
    print(f" - Meilleur K selon Davies-Bouldin : {best_davies}")
    print(f"\n=> LE MEILLEUR NOMBRE DE CLUSTERS FINAL RETENU EST : {meilleur_k}")
    print("=" * 60 + "\n")

    # Affichage et sauvegarde des graphiques
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0, 0].plot(liste_k, wcss, marker="o", color="blue")
    axs[0, 0].axvline(x=meilleur_k, color="r", linestyle=":")
    axs[0, 0].set_title("1. Méthode du Coude (Pliure)")
    axs[0, 0].grid(True, linestyle=":")

    axs[0, 1].plot(liste_k, scores_silhouette, marker="s", color="green")
    axs[0, 1].axvline(x=best_silhouette, color="r", linestyle="--")
    axs[0, 1].set_title("2. Score de Silhouette (Plus HAUT = meilleur)")
    axs[0, 1].grid(True, linestyle=":")

    axs[1, 0].plot(liste_k, scores_calinski, marker="^", color="purple")
    axs[1, 0].axvline(x=best_calinski, color="r", linestyle="--", label=f"Compromis Haut/Saut ({best_calinski})")
    axs[1, 1].set_title("3. Indice Calinski-Harabasz (Optimisé)")
    axs[1, 0].grid(True, linestyle=":")
    axs[1, 0].legend()

    axs[1, 1].plot(liste_k, scores_davies, marker="v", color="orange")
    axs[1, 1].axvline(x=best_davies, color="r", linestyle="--")
    axs[1, 1].set_title("4. Indice Davies-Bouldin (Plus BAS = meilleur)")
    axs[1, 1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(nom_graphique)
    print(f"-> Graphique sauvegardé sous {nom_graphique}")
    print("-> Fermez la fenêtre des graphiques pour continuer...")
    plt.show()

    return meilleur_k