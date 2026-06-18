import os
import webbrowser
import folium
import numpy as np


def verifier_coordonnees_france(lat, lon):
    """Vérifie si les coordonnées appartiennent à la France métropolitaine ou à la Corse"""
    # Zone globale englobant la France continentale et la Corse
    lat_min, lat_max = 41.3, 51.1
    lon_min, lon_max = -5.2, 9.6
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def localiser_point_utilisateur(kmeans_modele, couleurs_clusters, colonne_lat, colonne_lon):
    """Demande un point à l'utilisateur, trouve son cluster et affiche une carte allégée"""
    print("\n--- RECHERCHE DE CLUSTER VIA VOS COORDONNÉES ---")

    while True:
        try:
            user_lat = float(input("Entrez la latitude (min=41,3 max=51,1) : "))
            user_lon = float(input("Entrez la longitude (min=-5,2 max=9,6) : "))

            if verifier_coordonnees_france(user_lat, user_lon):
                break
            else:
                print(" Erreur : Ces coordonnées ne sont pas en France métropolitaine ni en Corse. Réessayez.")
        except ValueError:
            print("Erreur : Veuillez entrer des nombres valides.")

    # 1. Prediction du cluster le plus proche via les centroïdes du modèle K-Means
    point = np.array([[user_lat, user_lon]])
    cluster_predit = int(kmeans_modele.predict(point)[0])
    couleur_point = couleurs_clusters[cluster_predit % len(couleurs_clusters)]

    print(f"\n Votre point appartient au Cluster n°{cluster_predit}(Couleur : {couleur_point})")

    # 2. Création de la carte allégée (Uniquement le Pin et les centroïdes)
    carte_legere = folium.Map(location=[user_lat, user_lon], zoom_start=7)

    # Ajout du Pin de l'utilisateur
    folium.Marker(
        location=[user_lat, user_lon],
        popup=f"<b>Votre position</b><br>Cluster prédit : {cluster_predit}",
        icon=folium.Icon(color=couleur_point, icon="user")
    ).add_to(carte_legere)

    # Ajout de tous les centroïdes des clusters
    centroides = kmeans_modele.cluster_centers_
    for i, c in enumerate(centroides):
        couleur_cent = couleurs_clusters[i % len(couleurs_clusters)]
        folium.Marker(
            location=[c[0], c[1]],
            popup=f"<b>Centre du Cluster {i}</b>",
            icon=folium.Icon(color="blue" if i != cluster_predit else "red", icon="plug",prefix="fa")
        ).add_to(carte_legere)

    # Sauvegarde et ouverture
    nom_fichier = "carte_recherche_utilisateur.html"
    carte_legere.save(nom_fichier)
    webbrowser.open("file://" + os.path.realpath(nom_fichier))
    print(f"-> Carte de localisation générée dans {nom_fichier}")