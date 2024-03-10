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


