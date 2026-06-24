
import joblib as jb
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

#importation du modèle

model= jb.load("modèle_puissance.pkl")

#création de données d'implantation

data={'prise_type_combo_ccs':False,
    'implantation_station':"Voirie",
    'nom_operateur' : "Solero",
    'nom_amenageur' : "Solero",
    'created_at' :"01-08-2005",
    'prise_type_chademo' :False,
    'tarification' :0,
    'date_maj':"01-08-2005",
    'contact_operateur' :"076025616102",
    'prise_type_2':False,

}

data= pd.DataFrame(data, index=[0])

#fonction de prédiction de la puissance nominale

def predict(dataa):
    encoder = OrdinalEncoder()

    # conversion des données aux format numérique
    for i in dataa.keys():
        dataa[[i]] = encoder.fit_transform(dataa[[i]])

    # sélection des données utiles
    values = dataa[['prise_type_combo_ccs', 'implantation_station', 'nom_operateur', 'nom_amenageur', 'created_at', 'prise_type_chademo', 'tarification', 'date_maj', 'contact_operateur', 'prise_type_2']].copy()

    # prédiction de la puissance
    return model.predict(values)

if __name__ == "__main__":

    ypred= predict(data)

    for index in range(len(ypred)):
            print(f"puissance prédite: {ypred[index]}")

