
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

file_path = "BDD_nettoyé_IA.csv"

data=pd.read_csv(file_path,encoding="utf8",sep=",",low_memory=False)

data = data.dropna()
encoder = OrdinalEncoder()

dataY = data["puissance_nominale"].copy()
dataX= data.drop(columns="puissance_nominale")

for index in dataX.keys():
    dataX[[index]]= encoder.fit_transform(dataX[[index]])

dataX= pd.DataFrame(dataX)

Xtrain, Xtest, ytrain, ytest = train_test_split(dataX, dataY, test_size=0.2, random_state=42)
model= jb.load("modèle_puissance.pkl")

def predict(dataa):

    values = dataa[
        ['datagouv_dataset_id', 'implantation_station', 'nom_operateur', 'prise_type_combo_ccs', 'nom_amenageur',
         'datagouv_organization_or_owner', 'prise_type_chademo', 'prise_type_2', 'restriction_gabarit',
         'contact_amenageur']].copy()

    return model.predict(values)

if __name__ == "__main__":

  ypred= predict(Xtest)

for index in range(len(ypred)-1):
        print(f"puissance réelle {(ytest.values[index])} , puissance prédite: {ypred[index]}")

print(f"score RMSE:{root_mean_squared_error(y_pred=ypred,y_true=ytest)}")