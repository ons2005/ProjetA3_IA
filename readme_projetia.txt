================================================================================
                    README - CLUSTERING DES BORNES DE RECHARGE
================================================================================

PRÉREQUIS
----------
Python 3.8+

Installer les dépendances :
  pip install pandas scikit-learn folium matplotlib numpy

Placer le fichier de données dans le même dossier que les scripts :
  BDD_nettoyé_IA.csv  (doit contenir les colonnes : consolidated_latitude,
                        consolidated_longitude)

python main.py
---------------------------------
  Étape 1 — Chargement des données
    Le script lit BDD_nettoyé_IA.csv et nettoie les lignes sans coordonnées GPS.

  Étape 2 — Recherche du nombre optimal de clusters (K)
    Le script calcule automatiquement le meilleur K (entre 2 et 10) en comparant
    4 méthodes statistiques. Une fenêtre s'ouvre avec les graphiques d'analyse.
    --> Fermez la fenêtre pour continuer.
    Note : si figure_graph.png existe déjà dans le dossier, cette étape est
    sautée et le graphique existant est affiché directement.

  Étape 3 — Génération de la carte globale
    Une carte HTML de toutes les bornes de France s'ouvre dans votre navigateur,
    chaque borne étant colorée selon son cluster.
    Note : si carte_toutes_les_bornes.html existe déjà, elle est ouverte
    directement sans recalcul.

  Étape 4 — Saisie de votre position
    Le script vous demande vos coordonnées GPS. Entrez une latitude et une
    longitude situées en France métropolitaine ou en Corse.

  Étape 5 — Résultat
    Le script affiche dans le terminal le numéro du cluster auquel appartient
    votre position, puis ouvre une carte HTML personnalisée montrant votre
    position et les centres de chaque cluster.


python interface.py
------------
Le script principal à lancer est main.py. Les autres fichiers (recherche_cluster.py,
point_recherche.py) sont des modules appelés automatiquement, vous n'avez pas
à les lancer directement.

  python interface.py


CE QUE LE SCRIPT FAIT
---------------------------------
  Étape 1 — Chargement des données
    Le script lit BDD_nettoyé_IA.csv, nettoie les lignes sans coordonnées GPS
    et entraîne directement un modèle K-Means avec 5 clusters.

  Étape 2 — Saisie de votre position
    Le script vous demande vos coordonnées GPS. Entrez une latitude et une
    longitude situées en France métropolitaine ou en Corse.

  Étape 3 — Résultat
    Le script affiche dans le terminal le numéro du cluster auquel appartient
    votre position, puis ouvre une carte HTML personnalisée montrant votre
    position et les centres des 5 clusters.


EXEMPLE D'EXÉCUTION
--------------------

  $ python interface.py

  --- LANCEMENT RAPIDE ---
  Initialisation des 5 zones de recharge...

  --- RECHERCHE DE VOTRE POSITION ---
  Entrez la latitude (ex: 48.85) : 48.85
  Entrez la longitude (ex: 2.35) : 2.35

  -> Votre point appartient au Cluster n°2 (Couleur : blue)

   Terminé ! La carte a été générée et ouverte dans votre navigateur (carte_directe_utilisateur.html).


FICHIER GÉNÉRÉ
---------------
  carte_directe_utilisateur.html     Carte personnalisée avec votre position
                                     et les centres des 5 clusters


COORDONNÉES GPS VALIDES (France métropolitaine + Corse)
--------------------------------------------------------
  Latitude  : entre 41.3  et 51.1
  Longitude : entre -5.2  et 9.6

  Exemples :
    Paris          →  lat: 48.85   lon: 2.35
    Lyon           →  lat: 45.75   lon: 4.85
    Marseille      →  lat: 43.30   lon: 5.37
    Bordeaux       →  lat: 44.84   lon: -0.58
    Strasbourg     →  lat: 48.57   lon: 7.75

================================================================================
