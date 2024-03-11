################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                       ALEJANDRA AGUIRRE                      #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

###importar librerias
# Manipulación y análisis de datos:
import pandas as pd
import numpy as np

# Visualización de datos:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc


# Aprendizaje y estadísticas
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
from itertools import product
import math

# Preprocesamiento de datos y escalado:
from sklearn.preprocessing import StandardScaler
from collections import Counter

#Modelado y slección de características
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector

# Evaluación de modelos y clasificación
from sklearn import metrics
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error, r2_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




# Otras funcionalidades
import sys
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones


conn = sql.connect("db_empleados") 
df = pd.read_sql("SELECT * FROM all_employees", conn)
#Borramos la columna Index ya que no aporta nada relevante
df.drop('index', axis = 1, inplace = True)  

######################################################################
#                                                                    #
#   SELECCIÓN DE VARIABLES                                           #
#                                                                    #
#   Método Arbol de Desición                                         #
#                                                                    #
######################################################################



# REVISAREMOS NUEVAMENTE EL ÁRBOL DE DECISION:
# Prepararemos el teerreno para un arbol de decisión:
# Se eliminan variables nulas y se seleccionan las variables numericas
df_arbol = df.select_dtypes(include=['float64', 'int64']).dropna()

# Separamos las variables predictoras de la variable objetivo
X = df_arbol.drop("Attrition", axis=1)   # X contendrá todas las variables excepto "Attrition"
y = df_arbol["Attrition"] # y será la variable que estamos tratando de predecir (Attrition)


# Entrenamos un arbol dedecision para hacer predicciones sobre nuevas instancias de datos
# Creamos un modelo de árbol de decisión con una profundidad máxima de 3.
model = DecisionTreeClassifier(max_depth=3) #Limitamos el arbol a 3 para evitar el sobreajuste
model.fit(X, y) # Entrenamos el modelo con las variables predictoras (X) y la variable objetivo (y).

# Obtenemos la importancia de cada variable según el árbol de decisión.
importances = model.feature_importances_ #importancia de las columnas
feature_names = X.columns  #nombre de todas las columnas

# Organizamos la importancia de todas las variables en un DataFrame para una mejor visualización.
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

#Imprimimos la importancia de las variables según el árbol de decisión, ordenadas de mayor a menor importancia.
print("Importancia de las variables según el árbol de decisión:\n", feature_importance_df.sort_values(by="Importance", ascending=False))


# Importancia de las variables según el árbol de decisión:
#                      Feature  Importance
# 12        TotalWorkingYears    0.540448
# 9        NumCompaniesWorked    0.121857
# 1   EnvironmentSatisfaction    0.110722
# 4                       Age    0.097991
# 8             MonthlyIncome    0.089455
# 2           JobSatisfaction    0.039527
# 17           JobInvolvement    0.000000
# 16     YearsWithCurrManager    0.000000
# 15  YearsSinceLastPromotion    0.000000
# 14           YearsAtCompany    0.000000
# 13    TrainingTimesLastYear    0.000000
# 0                EmployeeID    0.000000
# 11         StockOptionLevel    0.000000
# 10        PercentSalaryHike    0.000000
# 7                  JobLevel    0.000000
# 6                 Education    0.000000
# 5          DistanceFromHome    0.000000
# 3           WorkLifeBalance    0.000000
# 18        PerformanceRating    0.000000




# Se convierten los datos a enteros
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].astype(float).astype(int) 
df['TotalWorkingYears'] = df['TotalWorkingYears'].astype(float).astype(int)

# Se eliminan columnas no relevantes.
del df['resignationReason']
del df['retirementDate']
del df['retirementType']
del df['InfoDate']
del df['DateSurvey']
del df['SurveyDate']
df.dtypes

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Columna: {column}, Tipo: {df[column].dtypes}, Valores Únicos: {unique_values}")


#En este paso, se realiza la codificación one-hot de las columnas categóricas especificadas en columns_to_encode. 
columns_to_encode = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'EducationField',
                     'BusinessTravel', 'Department', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                     'PerformanceRating','StockOptionLevel', 'WorkLifeBalance']

# La función pd.get_dummies crea columnas binarias para cada categoría en estas columnas. 
df_encoded = pd.get_dummies(df, columns=columns_to_encode)

# Después, el DataFrame se actualiza para contener estas nuevas columnas codificadas como enteros.
df = df_encoded.astype(int)



for column in df.columns:
    unique_values = df[column].unique()
    print(f"Columna: {column}, Tipo: {df[column].dtypes}, Valores Únicos: {unique_values}")

# Aquí, se normalizan (escalan) las columnas numéricas especificadas en ScalerList utilizando StandardScaler de 
# scikit-learn. Esto asegura que estas columnas tengan una media de cero y una desviación estándar de uno.
scaler = StandardScaler()

ScalerList = ['Age','DistanceFromHome', 'MonthlyIncome',
                'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears',
                'TrainingTimesLastYear', 'YearsAtCompany',
                'YearsSinceLastPromotion','YearsWithCurrManager']

for i in ScalerList:
    scaler = StandardScaler()  # Crea una instancia de StandardScaler para cada columna
    df[[i]] = scaler.fit_transform(df[[i]])
    
print(df[ScalerList].describe())

#            Age  DistanceFromHome  MonthlyIncome  NumCompaniesWorked  \
# count  3.528000e+04      3.528000e+04   3.528000e+04        3.528000e+04   
# mean  -2.718914e-17      4.531523e-17   4.914184e-17       -9.455777e-17   
# std    1.000014e+00      1.000014e+00   1.000014e+00        1.000014e+00   
# min   -2.072192e+00     -1.010909e+00  -1.167343e+00       -1.073523e+00   
# 25%   -7.581700e-01     -8.875151e-01  -7.632087e-01       -6.734351e-01   
# 50%   -1.011589e-01     -2.705440e-01  -3.365516e-01       -2.733477e-01   
# 75%    6.653541e-01      5.932157e-01   3.988370e-01        5.268271e-01   
# max    2.526886e+00      2.444129e+00   2.867626e+00        2.527264e+00   

#        PercentSalaryHike  TotalWorkingYears  TrainingTimesLastYear  \
# count       3.528000e+04       3.528000e+04           3.528000e+04   
# mean        2.332224e-16      -6.303851e-17           8.881784e-17   
# std         1.000014e+00       1.000014e+00           1.000014e+00   
# min        -1.150554e+00      -1.449387e+00          -2.171982e+00   
# 25%        -8.772324e-01      -6.781452e-01          -6.201892e-01   
# 50%        -3.305891e-01      -1.639837e-01           1.557071e-01   
# 75%         7.626976e-01       4.787182e-01           1.557071e-01   
# max         2.675949e+00       3.692227e+00           2.483396e+00   

#        YearsAtCompany  YearsSinceLastPromotion  YearsWithCurrManager  
# count    3.528000e+04             3.528000e+04          3.528000e+04  
# mean    -3.625218e-17             1.369527e-17         -2.134851e-17  
# std      1.000014e+00             1.000014e+00          1.000014e+00  
# min     -1.144294e+00            -6.791457e-01         -1.155935e+00  
# 25%     -6.544537e-01            -6.791457e-01         -5.952272e-01  
# 50%     -3.278933e-01            -3.687153e-01         -3.148735e-01  
# 75%      3.252275e-01             2.521455e-01          8.065415e-01  
# max      5.386914e+00             3.977310e+00          3.610079e+00  

######################################################################
#                                                                    #
#   SELECCIÓN DE VARIABLES                                           #
#                                                                    #
#   Método KBEST                                                     #
#                                                                    #
######################################################################

# Conectamos nuevamente a la base de datos y guardamos los resultados
conn = sql.connect("db_empleados")
df.to_sql("df", conn, if_exists="replace", index=False)

# Se dividen los datos en características (XKbest) y la variable objetivo (yKbest). Para nuestra 
# variable objetivo ATTRITION
XKbest = df.drop("Attrition", axis=1)  # Características
yKbest = df["Attrition"]  # Variable objetivo    

# Se utiliza la clase SelectKBest del módulo sklearn.feature_selection para seleccionar las mejores k características. 
# En este caso, se están seleccionando 16 características utilizando la puntuación de análisis de varianza (f_classif). 
# Las características seleccionadas se almacenan en X_best.
k_best = SelectKBest(score_func=f_classif, k=16)  # con k= 16, las 16 mejores características
X_best = k_best.fit_transform(XKbest, yKbest)

# Se crea un DataFrame llamado feature_scores que contiene dos columnas: "Feature" (característica) y "Score" (puntuación). 
# Las puntuaciones se obtienen del atributo k_best.scores_. Luego, el DataFrame se ordena en función de las puntuaciones de mayor a menor.
feature_scores = pd.DataFrame({'Feature': XKbest.columns, 'Score': k_best.scores_})
feature_scores.sort_values(by='Score', ascending=False, inplace=True)

# Se obtienen las características seleccionadas utilizando el método get_support() de k_best.
# Estas características se almacenan en selected_featuresKbest.
selected_featuresKbest = XKbest.columns[k_best.get_support()]

# Puntuaciones de características:
#                           Feature        Score
# 58           MaritalStatus_Single  1120.028070
# 6               TotalWorkingYears  1063.280870
# 1                             Age   917.417530
# 10           YearsWithCurrManager   882.245861
# 8                  YearsAtCompany   648.885070
print("Puntuaciones de características:")
print(feature_scores)


# Características seleccionadas:
# Index(['Age', 'TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager',
#        'EnvironmentSatisfaction_1.0', 'EducationField_Human Resources',
#        'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
#        'Department_Human Resources', 'JobSatisfaction_1', 'JobSatisfaction_4',
#        'MaritalStatus_Divorced', 'MaritalStatus_Married',
#        'MaritalStatus_Single', 'WorkLifeBalance_1.0', 'WorkLifeBalance_3.0'],
#       dtype='object')
print("\nCaracterísticas seleccionadas por KBest:")
print(selected_featuresKbest)


# Se crea un nuevo DataFrame llamado df_variables_kbest que contiene solo las características 
# seleccionadas por la técnica KBest. Estas características son aquellas que tienen las puntuaciones 
# más altas según el análisis de varianza.
df_variables_kbest = XKbest[selected_featuresKbest].copy()

# Se agrega la variable objetivo ("Attrition") al nuevo DataFrame df_variables_kbest. 
# Esto crea un DataFrame que contiene únicamente las características seleccionadas y la variable objetivo.
df_variables_kbest['Attrition'] = df['Attrition']

######################################################################
#                                                                    #
#   SELECCIÓN DE VARIABLES                                           #
#                                                                    #
#   Método LASSO                                                     #
#                                                                    #
######################################################################

X_lasso = df.drop("Attrition", axis=1)  # Características
y_lasso = df["Attrition"]  # Variable objetivo 


lasso_model = Lasso(alpha=0.01)  # El parámetro alpha controla la fuerza de la regularización


lasso_model.fit(X_lasso, y_lasso)

lasso_coefficients = pd.DataFrame({'Feature': X_lasso.columns, 'Coefficient': lasso_model.coef_})


selected_features_lasso = lasso_coefficients[lasso_coefficients['Coefficient'] != 0]['Feature']


print("Características seleccionadas por Lasso:")
print(selected_features_lasso)

# Características seleccionadas por Lasso:
# 0                           EmployeeID
# 1                                  Age
# 3                        MonthlyIncome
# 4                   NumCompaniesWorked
# 6                    TotalWorkingYears
# 7                TrainingTimesLastYear
# 9              YearsSinceLastPromotion
# 10                YearsWithCurrManager
# 16         EnvironmentSatisfaction_1.0
# 36    BusinessTravel_Travel_Frequently
# 52                   JobSatisfaction_1
# 55                   JobSatisfaction_4
# 58                MaritalStatus_Single
# 67                 WorkLifeBalance_3.0


######################################################################
#                                                                    #
#   SELECCIÓN DE VARIABLES                                           #
#                                                                    #
#   Método Wrapper                                                   #
#                                                                    #
######################################################################
# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k):
  rfe = RFE(model, n_features_to_select=k, step=1)
  fit = rfe.fit(X, y)
  X_new = fit.support_
  print("Num Features: %s" % (fit.n_features_))
  print("Selected Features: %s" % (fit.support_))
  print("Feature Ranking: %s" % (fit.ranking_))
  return X_new

X_wrapper = df.drop("Attrition", axis=1)  # Características
y_wrapper = df["Attrition"]  # Variable objetivo 

# Se crea un modelo de rregresión logística con un maximo de 10000 iteraciones
model = LogisticRegression(max_iter=10000)

# Esto asegura que todas las características tengan una media de cero y una desviación estándar de uno
scaler = StandardScaler()

X_wrapper_std = scaler.fit_transform(X_wrapper)

# Contiene las características seleccionadas después de aplicar el método RFE.
X_new_selected = recursive_feature_selection(X_wrapper_std, y_wrapper, model, 16)

# Es un DataFrame que contiene las características seleccionadas escaladas, con los nombres de las columnas provenientes del DataFrame original.
selected_features_df = pd.DataFrame(X_wrapper_std[:, X_new_selected], columns=X_wrapper.columns[X_new_selected])

# Se imprime el DataFrame selected_features_df que muestra las características seleccionadas después de aplicar el método RFE y escalarlas.
print(selected_features_df)

# Las 16 características seleccionadas por el método Recursive Feature Elimination (RFE) con un modelo de regresión logística son las siguientes:

# NumCompaniesWorked
# YearsAtCompany
# EnvironmentSatisfaction_1.0
# BusinessTravel_Non-Travel
# BusinessTravel_Travel_Frequently
# Department_Human Resources
# JobRole_Manufacturing Director
# JobSatisfaction_1
# JobSatisfaction_4
# MaritalStatus_Single
# WorkLifeBalance_1.0
# Age
# TotalWorkingYears
# TrainingTimesLastYear
# YearsWithCurrManager
# MaritalStatus_Single


#             Age  NumCompaniesWorked  TotalWorkingYears  TrainingTimesLastYear  \
# 0      1.541369           -0.673435          -1.320847               2.483396   
# 1      1.541369           -0.673435          -1.320847               2.483396   
# 2      1.541369           -0.673435          -1.320847               2.483396   
# 3      1.541369           -0.673435          -1.320847               2.483396   
# 4     -0.648668           -1.073523          -0.678145               0.155707   
# ...         ...                 ...                ...                    ...   
# 35275  0.555852           -1.073523          -0.163984              -0.620189   
# 35276  0.336849           -1.073523           1.249960               2.483396   
# 35277  0.336849           -1.073523           1.249960               2.483396   
# 35278  0.336849           -1.073523           1.249960               2.483396   
# 35279  0.336849           -1.073523           1.249960               2.483396   

#        YearsAtCompany  YearsSinceLastPromotion  YearsWithCurrManager  \
# 0           -0.981014                -0.679146             -1.155935   
# 1           -0.981014                -0.679146             -1.155935   
# 2           -0.981014                -0.679146             -1.155935   
# 3           -0.981014                -0.679146             -1.155935   
# 4           -0.327893                -0.368715             -0.034520   
# ...               ...                      ...                   ...   
# 35275        0.325228                 1.493867              1.086895   
# 35276        2.284590                 0.252146              1.367249   
# 35277        2.284590                 0.252146              1.367249   
# 35278        2.284590                 0.252146              1.367249   
# 35279        2.284590                 0.252146              1.367249   

#        EnvironmentSatisfaction_1.0  BusinessTravel_Non-Travel  \
# 0                        -0.486854                    -0.3371   
# 1                        -0.486854                    -0.3371   
# 2                        -0.486854                    -0.3371   
# 3                        -0.486854                    -0.3371   
# 4                        -0.486854                    -0.3371   
# ...                            ...                        ...   
# 35275                    -0.486854                    -0.3371   
# 35276                     2.054005                    -0.3371   
# 35277                     2.054005                    -0.3371   
# 35278                     2.054005                    -0.3371   
# 35279                     2.054005                    -0.3371   

#        BusinessTravel_Travel_Frequently  Department_Human Resources  \
# 0                             -0.481859                   -0.211604   
# 1                             -0.481859                   -0.211604   
# 2                             -0.481859                   -0.211604   
# 3                             -0.481859                   -0.211604   
# 4                              2.075297                   -0.211604   
# ...                                 ...                         ...   
# 35275                         -0.481859                   -0.211604   
# 35276                         -0.481859                   -0.211604   
# 35277                         -0.481859                   -0.211604   
# 35278                         -0.481859                   -0.211604   
# 35279                         -0.481859                   -0.211604   

#        JobRole_Manufacturing Director  JobSatisfaction_1  JobSatisfaction_4  \
# 0                           -0.330808          -0.492193           1.491993   
# 1                           -0.330808          -0.492193           1.491993   
# 2                           -0.330808          -0.492193           1.491993   
# 3                           -0.330808          -0.492193           1.491993   
# 4                           -0.330808          -0.492193          -0.670245   
# ...                               ...                ...                ...   
# 35275                       -0.330808           2.031725          -0.670245   
# 35276                       -0.330808          -0.492193          -0.670245   
# 35277                       -0.330808          -0.492193          -0.670245   
# 35278                       -0.330808          -0.492193          -0.670245   
# 35279                       -0.330808          -0.492193          -0.670245   

#        MaritalStatus_Single  WorkLifeBalance_1.0  
# 0                 -0.685565            -0.239375  
# 1                 -0.685565            -0.239375  
# 2                 -0.685565            -0.239375  
# 3                 -0.685565            -0.239375  
# 4                  1.458650            -0.239375  
# ...                     ...                  ...  
# 35275             -0.685565            -0.239375  
# 35276             -0.685565            -0.239375  
# 35277             -0.685565            -0.239375  
# 35278             -0.685565            -0.239375  
# 35279             -0.685565            -0.239375  

######################################################################
#                                                                    #
#  MODELO DE REGRESIÓN LOGÍSTICA                                     #
#                                                                    #
#   Con matriz de confusion                                          #
#                                                                    #
######################################################################


# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_wrapper_std[:, X_new_selected], y_wrapper, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión logística
model = LogisticRegression(max_iter=10000)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Imprimir resultados
# El 85.2% de las predicciones del modelo fueron correctas en el conjunto de prueba. Esta métrica se calcula como (verdaderos positivos + verdaderos negativos) / total.
print(f'Exactitud en el conjunto de prueba: {accuracy:.3f}\n')

print('Matriz de Confusión:')
print(conf_matrix)
print('\nReporte de Clasificación:')
print(classification_rep)


# Reporte de Clasificación:
#               precision    recall  f1-score   support

#            0       0.86      0.98      0.92      5919
#            1       0.66      0.16      0.26      1137

#     accuracy                           0.85      7056
#    macro avg       0.76      0.57      0.59      7056
# weighted avg       0.83      0.85      0.81      7056



###Matriz de confusión
matriz= confusion_matrix(y_test, y_pred)
matriz_display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=['No renuncia', 'renuncia'])
matriz_display.plot()
plt.show()


### metricas del modelo
tn, fp, fn, tp = matriz.ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
especificidad = tn / (fp + tn)
f1_score = 2*(precision*recall)/(precision+recall)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Especificidad: {especificidad}')
print(f'F1 score: {f1_score}')

# Precision: 0.6630434782608695
# Recall: 0.16094986807387862
# Especificidad: 0.9842878864673087
# F1 score: 0.2590233545647558


######################################################################
#                                                                    #
#           RANDOM FOREST CLASSIFIER                                 #
#                                                                    #
#   Con matriz de confusion                                          #
#                                                                    #
######################################################################
## se crea el modelo
X_train_modelo3=X_train
X_test_modelo3=X_test

ranfor = RandomForestClassifier(class_weight="balanced",n_estimators = 150,criterion= 'gini', max_depth= 5,max_leaf_nodes = 10,n_jobs= -1,random_state = 123)
ranfor.fit(X_train_modelo3, y_train)

###metricas
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor.predict(X_train_modelo3)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor.predict(X_train_modelo3)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor.predict(X_test_modelo3)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor.predict(X_test_modelo3)))


# Train - Accuracy : 0.7740221088435374
# Train - classification report:
#                precision    recall  f1-score   support

#            0       0.92      0.80      0.86     23673
#            1       0.38      0.65      0.48      4551

#     accuracy                           0.77     28224
#    macro avg       0.65      0.72      0.67     28224
# weighted avg       0.84      0.77      0.80     28224

# Test - Accuracy : 0.7769274376417233
# Test - classification report :               precision    recall  f1-score   support

#            0       0.92      0.80      0.86      5919
#            1       0.38      0.64      0.48      1137

#     accuracy                           0.78      7056
#    macro avg       0.65      0.72      0.67      7056
# weighted avg       0.83      0.78      0.80      7056


# Matriz de confusion
cm= confusion_matrix(y_test, ranfor.predict(X_test_modelo3))
# Visualización de la matriz de confusion
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['No renuncia', 'renuncia'])
cm_display.plot()
plt.show()


######################################################################
#                                                                    #
#           GRADIENT BOOSTING CLASSIFIER                             #
#                                                                    #
#   Con matriz de confusion                                          #
#                                                                    #
######################################################################
X_train_modelo4, X_test_modelo4, y_train_res, y_test = train_test_split(X_wrapper_std[:, X_new_selected], y_wrapper, test_size=0.2, random_state=42)

gboos = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_features=4, random_state=123)
gboos.fit(X_train_modelo4, y_train_res)
y_pred = gboos.predict(X_test_modelo4)

print("Test - Accuracy :", accuracy_score(y_test, y_pred))
print("Test - Classification report :\n", classification_report(y_test, y_pred))

# Test - Accuracy : 0.8802437641723356
# Test - Classification report :
#                precision    recall  f1-score   support

#            0       0.88      0.99      0.93      5919
#            1       0.84      0.32      0.46      1137

#     accuracy                           0.88      7056
#    macro avg       0.86      0.65      0.70      7056
# weighted avg       0.88      0.88      0.86      7056



cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No renuncia', 'Renuncia'])
cm_display.plot()
plt.show()

y_prob = gboos.predict_proba(X_test_modelo4)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()



######################################################################
#                                                                    #
#           SUPPORT VECTOR MACHINE                                   #
#                                                                    #
#   Con matriz de confusion                                          #
#                                                                    #
######################################################################

##se crea el modelo
X_train_modelo5 = X_train_modelo4  # Utilizando el conjunto de entrenamiento previamente definido
X_test_modelo5 = X_test_modelo4    # Utilizando el conjunto de prueba previamente definido


svm_model = SVC(C=1.5, kernel='linear', class_weight='balanced', max_iter=-1, random_state=123)
svm_model.fit(X_train_modelo5, y_train_res)  # Ajustar al conjunto de entrenamiento


print("Train - Accuracy:", accuracy_score(y_train_res, svm_model.predict(X_train_modelo5)))
print("Train - Classification Report:\n", classification_report(y_train_res, svm_model.predict(X_train_modelo5)))
print("Test - Accuracy:", accuracy_score(y_test, svm_model.predict(X_test_modelo5)))
print("Test - Classification Report:", classification_report(y_test, svm_model.predict(X_test_modelo5)))


# Train - Accuracy: 0.7048256802721088
# Train - Classification Report:
#                precision    recall  f1-score   support

#            0       0.93      0.70      0.80     23673
#            1       0.32      0.72      0.44      4551

#     accuracy                           0.70     28224
#    macro avg       0.62      0.71      0.62     28224
# weighted avg       0.83      0.70      0.74     28224

# Test - Accuracy: 0.7101757369614512
# Test - Classification Report:               precision    recall  f1-score   support

#            0       0.93      0.71      0.80      5919
#            1       0.32      0.73      0.45      1137

#     accuracy                           0.71      7056
#    macro avg       0.63      0.72      0.63      7056
# weighted avg       0.83      0.71      0.75      7056


# Matriz de Confusión
cm_svm = confusion_matrix(y_test, svm_model.predict(X_test_modelo5))
# Visualización de la Matriz de Confusión
cm_svm_display = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['No renuncia', 'renuncia'])
cm_svm_display.plot()
plt.show()


y_probs_svm = svm_model.decision_function(X_test_modelo5)
roc_auc_svm = roc_auc_score(y_test, y_probs_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_probs_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM (AUC = {:.2f})'.format(roc_auc_svm))
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - SVM')
plt.legend(loc='lower right')
plt.show()


