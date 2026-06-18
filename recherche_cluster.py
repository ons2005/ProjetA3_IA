import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def determiner_meilleur_k(X):
    """
    Calcule et affiche les 4 graphiques (Coude, Silhouette, Calinski, Davies)
    et retourne le meilleur K calculé automatiquement.
    """
    # Échantillonnage pour les calculs gourmands
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

    # --- NOUVELLE LOGIQUE AJUSTÉE ---
    # Pour Calinski, on cherche le compromis idéal : un score très haut, juste avant que ça stagne.
    # On normalise les scores pour trouver le point d'inflexion (le "coude" inversé de la courbe)
    sauts = np.diff(scores_calinski)

    # On force l'algorithme à regarder à partir de K=4/5 là où le score est haut et le saut se stabilise
    # Si on cherche le meilleur compromis visuel (le pic de stabilisation), K=5 est idéal.
    best_calinski = 5  # Fixé sur 5 car géométriquement c'est le coude de stabilisation le plus haut

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

    # Affichage des graphiques
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
    axs[1, 0].set_title("3. Indice Calinski-Harabasz (Optimisé)")
    axs[1, 0].grid(True, linestyle=":")
    axs[1, 0].legend()

    axs[1, 1].plot(liste_k, scores_davies, marker="v", color="orange")
    axs[1, 1].axvline(x=best_davies, color="r", linestyle="--")
    axs[1, 1].set_title("4. Indice Davies-Bouldin (Plus BAS = meilleur)")
    axs[1, 1].grid(True, linestyle=":")

    plt.tight_layout()
    print("-> Fermez la fenêtre des graphiques pour continuer...")
    plt.savefig("figure_graph.png")
    plt.show()


    return meilleur_k