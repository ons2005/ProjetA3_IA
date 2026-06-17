import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


file_path = "Besoin_Client_4/BDD_nettoyé_IA.csv"

data=pd.read_csv(file_path,encoding="utf8",sep=",")

#associe des données numériques à chaque type d'implantation
encoder= OrdinalEncoder()

implantation_cat= data[["implantation_station"]]

data[["implantation_station"]] = encoder.fit_transform(implantation_cat)

data[["prise_type_ef"]] = data[["prise_type_ef"]].fillna(data[["prise_type_ef"]].median())
data[["prise_type_2"]] = data[["prise_type_2"]].fillna(data[["prise_type_2"]].median())
data[["prise_type_autre"]] = data[["prise_type_autre"]].fillna(data[["prise_type_autre"]].median())
data[["prise_type_combo_ccs"]] = data[["prise_type_combo_ccs"]].fillna(data[["prise_type_combo_ccs"]].median())
data[["prise_type_chademo"]] = data[["prise_type_chademo"]].fillna(data[["prise_type_chademo"]].median())


#sélectionne les données X et Y
dataX= data[["implantation_station","prise_type_ef","prise_type_2","prise_type_combo_ccs","prise_type_chademo","prise_type_autre"]].copy()

dataY= data["puissance_nominale"].copy()


# divise la base données en base d'apprentissage, de validation et de test

Xtrain, Xvalidtest = train_test_split(dataX,train_size=0.6)
Xvalid, Xtest = train_test_split(Xvalidtest,train_size=0.5)

ytrain, yvalidtest = train_test_split(dataY,train_size=0.6)
yvalid, ytest = train_test_split(yvalidtest,train_size=0.5)

forest= RandomForestRegressor()

forest.fit(X=Xtrain,y=ytrain)

ypred= forest.predict(X=Xtest)

print(pow(mean_squared_error(y_true=ytest, y_pred=ypred),exp=0.5))


param_grid = {
    'max_features': [1.0, 'sqrt', 'log2'],
    'max_depth': [None,4, 5, 6, 7, 8],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'n_estimators': [1,10,100]
}

grid= GridSearchCV(estimator=forest,param_grid=param_grid,scoring="r2")

# Fit the model
grid.fit(Xtrain, ytrain)

# Get the best parameters and score
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)

# Predict on the test set
y_pred = grid.predict(Xtest)

# Evaluate the model
score = r2_score(ytest, ypred)
print("score r2: ", score)




