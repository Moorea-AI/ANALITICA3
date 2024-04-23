################################################################
#               SEGUNDO TRABAJO PARA LA MATERIA:               #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE SALUD                             #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################


import pandas as pd # Librería para manipulación y análisis de datos
import numpy as np # Librería para operaciones numéricas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


from b_EDA import data, stat_data,  X_train, X_test, y_train, y_test

# La idea del presente trabajo es poder predecir la tasa de suicidios al año
# con base en los datos recolectados hasta 2016
# Regresion lineal, KNN, random forest, árboles de decisión, gradient boosting
# E investigando descubrí uno que se llama SuperLearner que voy a intentar

# Almacenaremos los resultados
ML_Model = []
acc_train = []
acc_test = []
rmse_train = []
rmse_test = []

def storeResults(model, a,b,c,d):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))
  rmse_train.append(round(c, 3))
  rmse_test.append(round(d, 3))
  

knn = KNeighborsRegressor()

param_grid = {'n_neighbors':list(range(1, 31)), 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(knn, param_grid , cv=10)
knn_grid.fit(X_train, y_train)

knn_para = knn_grid.best_params_
print(knn_para)


y_train_knn = knn_grid.predict(X_train)
y_test_knn = knn_grid.predict(X_test)


#mediremos el accuracy del modelo
acc_train_knn = knn_grid.score(X_train, y_train) 
acc_test_knn = knn_grid.score(X_test, y_test)

# y RMSE
rmse_train_knn = np.sqrt(mean_squared_error(y_train, y_train_knn))
rmse_test_knn = np.sqrt(mean_squared_error(y_test, y_test_knn))

print("KNN: Accuracy de los daos de entrenamiento: {:.3f}".format(acc_train_knn))
print("KNN: Accuracy de los datos de prueba: {:.3f}".format(acc_test_knn))
print('\nKNN: RMSE de los datos de entrenamiento:', rmse_train_knn)
print('KNN: RMSE de los datos de prueba:', rmse_test_knn)


storeResults('k-Nearest Neighbors Regression', acc_train_knn, acc_test_knn, rmse_train_knn, rmse_test_knn)

