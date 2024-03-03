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

#### Cargar paquetes siempre al inicio
import sqlite3 as sql #### para bases de datos sql
import a_funciones as a_funciones ### archivo de funciones propias
import seaborn as sns
import numpy as np 
import graphviz

import pandas as pd ### para manejo de datos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones

import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos

from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import export_text ## para exportar reglas del árbol
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import  KNeighborsClassifier


conn = sql.connect("db_empleados") 
df = pd.read_sql("SELECT * FROM all_employees", conn)
df.drop('index', axis = 1, inplace = True)  
#Borramos la columna Index ya que no aporta nada relevante

df['NumCompaniesWorked'] = df['NumCompaniesWorked'].astype(float).astype(int) 
df['TotalWorkingYears'] = df['TotalWorkingYears'].astype(float).astype(int)
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



columns_to_encode = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'EducationField',
                     'BusinessTravel', 'Department', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                     'PerformanceRating','StockOptionLevel', 'WorkLifeBalance']


df_encoded = pd.get_dummies(df, columns=columns_to_encode)
df = df_encoded.astype(int)

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Columna: {column}, Tipo: {df[column].dtypes}, Valores Únicos: {unique_values}")

scaler = StandardScaler()

ScalerList = ['Age','DistanceFromHome', 'MonthlyIncome',
                'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears',
                'TrainingTimesLastYear', 'YearsAtCompany',
                'YearsSinceLastPromotion','YearsWithCurrManager']

for i in ScalerList:
    scaler = StandardScaler()  # Crea una instancia de StandardScaler para cada columna
    df[[i]] = scaler.fit_transform(df[[i]])
    
print(df[ScalerList].describe())

conn = sql.connect("db_empleados")
df.to_sql("df", conn, if_exists="replace", index=False)


XKbest = df.drop("Attrition", axis=1)  # Características
yKbest = df["Attrition"]  # Variable objetivo    

k_best = SelectKBest(score_func=f_classif, k=16)  # con k= 16, las 16 mejores características
X_best = k_best.fit_transform(XKbest, yKbest)

feature_scores = pd.DataFrame({'Feature': XKbest.columns, 'Score': k_best.scores_})
feature_scores.sort_values(by='Score', ascending=False, inplace=True)

selected_featuresKbest = XKbest.columns[k_best.get_support()]


print("Puntuaciones de características:")
print(feature_scores)
print("\nCaracterísticas seleccionadas:")
print(selected_featuresKbest)


df_variables_kbest = XKbest[selected_featuresKbest].copy()


df_variables_kbest['Attrition'] = df['Attrition']


Xsfs = df.drop("Attrition", axis=1)  # Características
ysfs = df["Attrition"]  # Variable objetivo

sfs = SequentialFeatureSelector(LogisticRegression(class_weight="balanced", max_iter=500), 
                                n_features_to_select=16, 
                                direction= "forward",  
                                scoring='f1')

sfs.fit(Xsfs, ysfs)


selected_featuresSFS = Xsfs.columns[sfs.get_support()]
print("Características seleccionadas:", selected_featuresSFS)

df_variables_sfs = Xsfs[selected_featuresSFS].copy()


df_variables_sfs['Attrition'] = df['Attrition']


print("Variables con KBest:",sorted(selected_featuresKbest))
print("Variables con SFS:",sorted(selected_featuresSFS))


conn = sql.connect("db_empleados")
df_variables_kbest.to_sql("df_variables_kbest", conn, if_exists = "replace", index=False)### Llevar tablas a base de datos
df_variables_sfs.to_sql("df_variables_sfs", conn, if_exists = "replace", index=False) ### Llevar tablas a base de datos


knn_scores = []
tree_scores = []
num_features_list = range(1, len(XKbest.columns) + 1)  # Prueba desde 1 hasta el número máximo de características


for num_features in num_features_list:
    X_knn = XKbest.iloc[:, :num_features]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_scores.append(np.mean(cross_val_score(knn, X_knn, yKbest, cv=5)))  # Validación cruzada de 5-fold



for num_features in num_features_list:
    X_tree = XKbest.iloc[:, :num_features]
    tree = DecisionTreeClassifier(random_state=42)
    tree_scores.append(np.mean(cross_val_score(tree, X_tree, yKbest, cv=5)))  # Validación cruzada de 5-fold


best_num_features_knn = num_features_list[np.argmax(knn_scores)]
best_num_features_tree = num_features_list[np.argmax(tree_scores)]


print("Número óptimo de características para K Nearest Neighbors:", best_num_features_knn)
print("Número óptimo de características para Árbol de Decisión:", best_num_features_tree)

