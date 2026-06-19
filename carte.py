import os
import webbrowser
import folium
import pandas as pd
from sklearn.cluster import KMeans

# 1. Chargement de l'intégralité des données
df = pd.read_csv("BDD_nettoyé_IA.csv", sep=",", encoding="utf8", low_memory=False)

colonne_lat = "consolidated_latitude"
colonne_lon = "consolidated_longitude"

# Nettoyage : on supprime uniquement les lignes sans coordonnées géographiques
df = df.dropna(subset=[colonne_lat, colonne_lon])

# 2. Préparation des données pour le K-Means
X = df[[colonne_lat, colonne_lon]]

# 3. Application de l'algorithme K-Means
nb_clusters = 5
kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# 4. Création de la carte Folium centrée sur la France
carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

# Liste de couleurs pour vos 5 clusters
couleurs_clusters = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "pink",
    "gray", "black", "lightred", "lightblue", "lightgreen"
    ]

# 5. Ajout de TOUTES les bornes de recharge sur la carte
print(f"Génération de la carte pour {len(df)} bornes de recharge...")

for _, ligne in df.iterrows():
    cluster_id = int(ligne["cluster"])
    couleur = couleurs_clusters[cluster_id % len(couleurs_clusters)]

    # Création d'un marqueur (points plus petits pour gérer la masse de données)
    folium.CircleMarker(
        location=[ligne[colonne_lat], ligne[colonne_lon]],
        radius=1,  # Plus petit pour éviter les gros pâtés de couleur
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.6,
        popup=f"Cluster: {cluster_id}",
    ).add_to(carte)

# 6. Sauvegarde et ouverture automatique
nom_fichier_carte = "carte_toutes_les_bornes.html"
carte.save(nom_fichier_carte)

# Ouverture automatique dans le navigateur
webbrowser.open("file://" + os.path.realpath(nom_fichier_carte))
print(f"Terminé ! La carte complète a été ouverte dans votre navigateur.")