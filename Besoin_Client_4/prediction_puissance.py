import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

file_path = "BDD_nettoyé_IA.csv"

data=pd.read_csv(file_path,encoding="utf8",sep=",")

#associe des données numériques à chaque type d'implantation
encoder= OrdinalEncoder()

implantation_cat= data[["implantation_station"]]

data[["implantation_station"]] = encoder.fit_transform(implantation_cat)


#sélectionne les données X et Y
dataX= data[["implantation_station","prise_type_ef","prise_type_2","prise_type_combo_ccs","prise_type_chademo","prise_type_autre"]].copy()

dataY= data["puissance_nominale"].copy()


# divise la base données en base d'apprentissage et de test
train_setX, test_setX = train_test_split(dataX,test_size=0.2,random_state=42)

train_setY, test_setY = train_test_split(dataY,test_size=0.2,random_state=42)


tree= DecisionTreeRegressor()

tree.fit(X=dataX,y=dataY)


