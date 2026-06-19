import os
import webbrowser
import folium
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Palette de couleurs pour les 5 clusters
COULEURS_CLUSTERS = ["red", "blue", "green", "purple", "orange"]


def main():
    print("--- LANCEMENT RAPIDE ---")

    # 1. Chargement et entraînement flash (K=5 imposé)
    print("Initialisation des 5 zones de recharge...")
    df = pd.read_csv("BDD_nettoyé_IA.csv", sep=",", encoding="utf8", low_memory=False)

    colonne_lat = "consolidated_latitude"
    colonne_lon = "consolidated_longitude"
    df = df.dropna(subset=[colonne_lat, colonne_lon])
    X = df[[colonne_lat, colonne_lon]]

    # On va droit au but : entraînement direct avec 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X)
    centroides = kmeans.cluster_centers_

    # 2. Demande immédiate des coordonnées à l'utilisateur
    print("\n--- RECHERCHE DE VOTRE POSITION ---")
    while True:
        try:
            user_lat = float(input("Entrez la latitude (ex: 48.85) : "))
            user_lon = float(input("Entrez la longitude (ex: 2.35) : "))

            # Vérification si on est bien en France
            if 41.3 <= user_lat <= 51.1 and -5.2 <= user_lon <= 9.6:
                break

            print(" Erreur : Les coordonnées doivent être en France (Lat: 41.3 à 51.1 / Lon: -5.2 à 9.6).")
        except ValueError:
            print(" Erreur : Veuillez entrer des nombres valides.")

    # 3. Prédiction du cluster pour le point utilisateur
    point = np.array([[user_lat, user_lon]])
    cluster_predit = int(kmeans.predict(point)[0])
    couleur_point = COULEURS_CLUSTERS[cluster_predit]

    print(f"\n-> Votre point appartient au Cluster n°{cluster_predit} (Couleur : {couleur_point})")

    # 4. Création et affichage de la carte
    carte = folium.Map(location=[user_lat, user_lon], zoom_start=7)

    # Marqueur de l'utilisateur
    folium.Marker(
        location=[user_lat, user_lon],
        popup=f"<b>Votre position</b><br>Cluster : {cluster_predit}",
        icon=folium.Icon(color="light_green", icon="user")
    ).add_to(carte)

    # Ajout des 5 centroïdes sur la carte
    for i, c in enumerate(centroides):
        folium.Marker(
            location=[c[0], c[1]],
            popup=f"<b>Centre du Cluster {i}</b>",
            icon=folium.Icon(color="blue" if i != cluster_predit else "red", icon="plug", prefix="fa")
        ).add_to(carte)

    # Sauvegarde et ouverture automatique du navigateur
    nom_fichier = "carte_directe_utilisateur.html"
    carte.save(nom_fichier)
    webbrowser.open("file://" + os.path.realpath(nom_fichier))

    print(f"\n Terminé ! La carte a été générée et ouverte dans votre navigateur ({nom_fichier}).")


if __name__ == "__main__":
    main()