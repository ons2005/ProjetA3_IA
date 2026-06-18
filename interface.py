import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

# Importation de vos modules
from recherche_cluster import determiner_meilleur_k
from point_recherche import verifier_coordonnees_france

# --- Configuration ---
st.set_page_config(page_title="Bornes Recharge", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("BDD_nettoyé_IA.csv", sep=",", encoding="utf8", low_memory=False)
    df = df.dropna(subset=["consolidated_latitude", "consolidated_longitude"])
    return df


df = load_data()
col_lat, col_lon = "consolidated_latitude", "consolidated_longitude"
X = df[[col_lat, col_lon]]

# --- Menu ---
menu = st.sidebar.radio("Navigation", ["1. Analyse", "2. Carte complète", "3. Recherche de point"])

# --- Fonctionnalités ---

if menu == "1. Analyse":
    st.header("Analyse des Clusters")
    if st.button("Lancer l'analyse (Graphiques)"):
        # Note: ceci affichera les graphiques dans la console/fenêtre locale Matplotlib
        k = determiner_meilleur_k(X)
        st.write(f"Meilleur K calculé : {k}")

elif menu == "2. Carte complète":
    st.header("Visualisation complète")
    nb_clusters = st.slider("Nombre de clusters", 2, 15, 5)

    if st.button("Générer la carte"):
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init=10).fit(X)
        df["cluster"] = kmeans.labels_

        # Création de la carte
        m = folium.Map(location=[46.6, 1.8], zoom_start=6)
        for _, ligne in df.iterrows():
            folium.CircleMarker(
                location=[ligne[col_lat], ligne[col_lon]],
                radius=1,
                color="blue",
                popup=f"Cluster: {int(ligne['cluster'])}"
            ).add_to(m)
        st_folium(m, width=800, height=500)

elif menu == "3. Recherche de point":
    st.header("Recherche par coordonnées")
    user_lat = st.number_input("Latitude", value=46.6033)
    user_lon = st.number_input("Longitude", value=1.8883)
    nb_clusters = st.slider("Nombre de clusters pour le modèle", 2, 15, 5)

    if st.button("Localiser"):
        if verifier_coordonnees_france(user_lat, user_lon):
            kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init=10).fit(X)
            cluster_predit = kmeans.predict([[user_lat, user_lon]])[0]
            st.success(f"Le point appartient au cluster n°{cluster_predit}")

            # Affichage carte simplifiée
            m = folium.Map(location=[user_lat, user_lon], zoom_start=8)
            folium.Marker([user_lat, user_lon], popup="Votre position").add_to(m)
            st_folium(m, width=800, height=500)
        else:
            st.error("Coordonnées hors de France.")