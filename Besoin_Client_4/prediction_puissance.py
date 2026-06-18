
import joblib as jb
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

#importation du modèle

model= jb.load("modèle_puissance.pkl")

#création de données d'implantation

data={
    "datagouv_dataset_id": "6740a4d4ca0ce2aba15a3c68",
    "implantation_station": "Voirie",
    "nom_operateur": "SOLVEO ENERGIES",
    "prise_type_combo_ccs": True,
    "nom_amenageur": "SOLVEO ENERGIES",
    "datagouv_organization_or_owner": "solveo-energies",
    "prise_type_chademo": True,
    "prise_type_2": False,
    "restriction_gabarit": "max 2m",
    "contact_amenageur": "t.prouin@sde23.fr"
}

data= pd.DataFrame(data, index=[0])

#fonction de prédiction de la puissance nominale

def predict(dataa):
    encoder = OrdinalEncoder()

    # conversion des données aux format numérique
    for i in dataa.keys():
        dataa[[i]] = encoder.fit_transform(dataa[[i]])

    # sélection des données utiles
    values = dataa[["datagouv_dataset_id", "implantation_station", "nom_operateur", "prise_type_combo_ccs", "nom_amenageur",
         "datagouv_organization_or_owner", "prise_type_chademo", "prise_type_2", "restriction_gabarit",
         "contact_amenageur"]].copy()

    # prédiction de la puissance
    return model.predict(values)

if __name__ == "__main__":

    ypred= predict(data)

    for index in range(len(ypred)):
            print(f"puissance prédite: {ypred[index]}")

