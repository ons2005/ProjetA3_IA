import os
import webbrowser
import folium
import pandas as pd
from sklearn.cluster import KMeans

# Importation de nos fonctions créées dans les autres fichiers
from recherche_cluster import determiner_meilleur_k
from point_recherche import localiser_point_utilisateur

# Palette globale de couleurs (15 couleurs)
COULEURS_CLUSTERS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "pink",
    "gray", "black", "lightred", "lightblue", "lightgreen"
]


def generer_carte_complete(df, colonne_lat, colonne_lon):
    """Affiche la carte de France complète avec toutes les données de la BDD"""
    print(f"\nGénération de la carte globale ({len(df)} bornes de recharge)...")
    carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

    for _, ligne in df.iterrows():
        cluster_id = int(ligne["cluster"])
        couleur = COULEURS_CLUSTERS[cluster_id % len(COULEURS_CLUSTERS)]

        folium.CircleMarker(
            location=[ligne[colonne_lat], ligne[colonne_lon]],
            radius=2,
            color=couleur,
            fill=True,
            fill_color=couleur,
            fill_opacity=0.6,
            popup=f"Cluster: {cluster_id}",
        ).add_to(carte)

    nom_fichier = "carte_toutes_les_bornes.html"
    carte.save(nom_fichier)
    webbrowser.open("file://" + os.path.realpath(nom_fichier))
    print(f"-> Carte complète ouverte dans votre navigateur ({nom_fichier}).")


def main():
    # 1. Chargement et nettoyage des données
    print("--- DÉMARRAGE DU PROGRAMME ---")
    df = pd.read_csv("BDD_nettoyé_IA.csv", sep=",", encoding="utf8", low_memory=False)

    colonne_lat = "consolidated_latitude"
    colonne_lon = "consolidated_longitude"
    df = df.dropna(subset=[colonne_lat, colonne_lon])
    X = df[[colonne_lat, colonne_lon]]

    # 2. Étape d'analyse des modèles pour trouver le K idéal
    meilleur_k = determiner_meilleur_k(X)

    # 3. Application du modèle K-Means final avec le K optimal trouvé
    print(f"\nCalcul du clustering final avec K = {meilleur_k}...")
    kmeans_final = KMeans(n_clusters=meilleur_k, random_state=42, n_init=10)
    df["cluster"] = kmeans_final.fit_predict(X)

    # 4. Génération de la carte complète (toutes les données)
    generer_carte_complete(df, colonne_lat, colonne_lon)

    # 5. Lancement du module interactif pour l'utilisateur
    localiser_point_utilisateur(kmeans_final, COULEURS_CLUSTERS, colonne_lat, colonne_lon)


if __name__ == "__main__":
    main()