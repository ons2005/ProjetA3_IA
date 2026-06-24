import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 1. Chargement des fichiers du modèle
try:
    best_model = joblib.load("modele_implantation_lgbm.pkl")
    encoder_y = joblib.load("label_encoder.pkl")
    kmeans = joblib.load("kmeans_zones.pkl")
    features_finales = joblib.load("features_liste.pkl")
    print("[Succès] Modèle et artefacts chargés.\n")
except FileNotFoundError as e:
    print(f"[Erreur] Fichier de sauvegarde introuvable : {e}")
    print("Vérifiez que les fichiers .pkl sont bien dans le même dossier.")
    exit()


def predire_nouvelle_borne(puissance, nbre_pdc, latitude, longitude,
                           code_postal, code_insee,
                           prise_type_2=1, prise_combo_ccs=0, prise_chademo=0,
                           paiement_cb=1, accessibilite_pmr=0, station_deux_roues=0):
    # Calcul de la zone géographique
    zone = kmeans.predict([[latitude, longitude]])[0]

    # Statistiques de repli par défaut (pour éviter de charger le gros CSV)
    p_moy_commune = 22.0
    pdc_moy_commune = 2.0
    nb_bornes_commune = 1
    dept = code_postal // 1000
    p_moy_dept = 22.0
    pdc_moy_dept = 2.0

    # Reconstruction du dictionnaire de features
    nouvelle_borne = {
        "puissance_nominale": [puissance],
        "nbre_pdc": [nbre_pdc],
        "accessibilite_pmr": [accessibilite_pmr],
        "prise_type_2": [prise_type_2],
        "paiement_cb": [paiement_cb],
        "puissance_par_pdc": [puissance / (nbre_pdc + 1)],
        "puissance_x_pdc": [puissance * nbre_pdc],
        "categorie_puissance": [pd.Series(
            pd.cut([puissance], bins=[0, 3.7, 7.4, 22, 50, 150, 9999], labels=[0, 1, 2, 3, 4, 5],
                   include_lowest=True).astype(float)).fillna(6).astype(int)[0]],
        "grande_station": [int(nbre_pdc >= 4)],
        "nb_types_prise": [prise_type_2 + prise_combo_ccs + prise_chademo],
        "borne_rapide": [int(prise_combo_ccs == 1 or prise_chademo == 1)],
        "consolidated_latitude": [latitude],
        "consolidated_longitude": [longitude],
        "consolidated_code_postal": [code_postal],
        "code_insee_commune": [code_insee],
        "departement": [dept],
        "zone_geo": [zone],
        "puissance_moy_commune": [p_moy_commune],
        "nb_pdc_moy_commune": [pdc_moy_commune],
        "nb_bornes_commune": [nb_bornes_commune],
        "puissance_moy_dept": [p_moy_dept],
        "nb_pdc_moy_dept": [pdc_moy_dept]
    }

    if "station_deux_roues" in features_finales:
        nouvelle_borne["station_deux_roues"] = [station_deux_roues]

    X_new = pd.DataFrame(nouvelle_borne)[features_finales]

    pred = best_model.predict(X_new)
    proba = best_model.predict_proba(X_new)

    label = encoder_y.inverse_transform(pred)[0]
    confiance = proba[0].max() * 100

    print("\n" + "=" * 40)
    print("        RÉSULTAT DE LA PRÉDICTION")
    print("=" * 40)
    print(f"Implantation prédite : {label}")
    print(f"Confiance            : {confiance:.1f}%")
    print("\nProbabilités par classe :")
    for cls, p in zip(encoder_y.classes_, proba[0]):
        print(f"  - {cls:25s}: {p * 100:.1f}%")
    print("=" * 40)


def demander_oui_non(question):
    """Demande une validation Oui/Non à l'utilisateur et renvoie 1 ou 0"""
    while True:
        reponse = input(f"{question} (o/n) : ").strip().lower()
        if reponse in ['o', 'oui']:
            return 1
        if reponse in ['n', 'non']:
            return 0
        print("Réponse invalide. Veuillez taper 'o' pour Oui ou 'n' pour Non.")


if __name__ == "__main__":
    print("=" * 50)
    print("   SAISIE MANUELLE D'UNE NOUVELLE BORNE")
    print("=" * 50)

    try:
        # Saisie des caractéristiques techniques
        puissance = float(input("Puissance nominale de la borne (en kW, 0.1 à 400.0) : "))
        nbre_pdc = int(input("Nombre de points de charge (pdc, 1 à 30) : "))

        # Saisie des caractéristiques géographiques
        latitude = float(input("Latitude (de 41.3 à 51.1) : "))
        longitude = float(input("Longitude (de -4.8 à 8.2) : "))
        code_postal = int(input("Code postal (de 01000 à 95880) : "))
        code_insee = int(input("Code INSEE de la commune (de 01001 à 95690) : "))

        print("\n--- Options supplémentaires ---")
        prise_type_2 = demander_oui_non("Dispose d'une prise Type 2 ?")
        prise_combo_ccs = demander_oui_non("Dispose d'une prise Combo CCS (recharge rapide) ?")
        prise_chademo = demander_oui_non("Dispose d'une prise CHAdeMO (recharge rapide) ?")
        paiement_cb = demander_oui_non("Permet le paiement par CB ?")
        accessibilite_pmr = demander_oui_non("Est accessible aux PMR ?")

        station_deux_roues = 0
        if "station_deux_roues" in features_finales:
            station_deux_roues = demander_oui_non("Est-ce une station dédiée aux deux-roues ?")

        # Lancement de la prédiction avec les données saisies
        predire_nouvelle_borne(
            puissance=puissance, nbre_pdc=nbre_pdc,
            latitude=latitude, longitude=longitude,
            code_postal=code_postal, code_insee=code_insee,
            prise_type_2=prise_type_2, prise_combo_ccs=prise_combo_ccs, prise_chademo=prise_chademo,
            paiement_cb=paiement_cb, accessibilite_pmr=accessibilite_pmr, station_deux_roues=station_deux_roues
        )

    except ValueError:
        print("\n[Erreur] Format de saisie incorrect. Assurez-vous d'entrer des nombres là où c'est demandé.")