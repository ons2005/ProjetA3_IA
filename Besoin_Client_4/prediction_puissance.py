import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt


file_path = "Besoin_Client_4/BDD_nettoyé_IA.csv"

data=pd.read_csv(file_path,encoding="utf8",sep=",")

#associe des données numériques à chaque type d'implantation
encoder= OrdinalEncoder()

implantation_cat= data[["implantation_station"]]

data[["implantation_station"]] = encoder.fit_transform(implantation_cat)


#sélectionne les données X et Y
dataX= data[["implantation_station","prise_type_ef","prise_type_2","prise_type_combo_ccs","prise_type_chademo","prise_type_autre"]].copy()

dataY= data["puissance_nominale"].copy()


# divise la base données en base d'apprentissage, de validation et de test

Xtrain, Xvalidtest = train_test_split(dataX,train_size=0.6)
Xvalid, Xtest = train_test_split(Xvalidtest,train_size=0.5)

ytrain, yvalidtest = train_test_split(dataY,train_size=0.6)
yvalid, ytest = train_test_split(yvalidtest,train_size=0.5)

tree= DecisionTreeRegressor()

tree.fit(X=Xtrain,y=ytrain)

ypred= tree.predict(X=Xtest)

print(pow(mean_squared_error(y_true=ytest, y_pred=ypred),exp=0.5))


param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None,4, 5, 6, 7, 8],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'splitter': ['best','random']
}

grid= GridSearchCV(estimator=tree,param_grid=param_grid,scoring="neg_root_mean_squared_error")

# Fit the model
grid.fit(Xtrain, ytrain)

# Get the best parameters and score
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)

# Predict on the test set
y_pred = grid.predict(Xtest)

# Evaluate the model
score = root_mean_squared_error(ytest, ypred)
print("score RMSE: ", score)




