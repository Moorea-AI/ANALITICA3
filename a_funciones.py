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


### Este es el archivo de funciones donde se recopilan todas las que se usarán y serán llamadas 
# desde otros archivos. Usaremos una matriz de confusión y una de correlación para ver la cohesión de 
#las variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Para poder ver gráficamente las variables
import seaborn as sns #Para poder ver gráficamente las variables.

import os  #Paquete OS: https://docs.python.org/es/3.10/library/os.html

from pathlib import Path

#Ya que es un problema de clasificación, quiero usar la matriz de confusion para 
# visualizar mejor la relación de las variables que influyen si un empleado se queda o no.
from sklearn.metrics import confusion_matrix 
#Se toman del archivo del profe por si nos sirven de algo más adelante:
from sklearn.impute import SimpleImputer 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 
from sklearn.feature_selection import RFE


#Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas
#En el archivo de SQL pondremos todas las consultas necesarias para construir la base de datos del modelo
# def ejecutar_sql(nombre_archivo, cur):
#     try:
#         # Mirar si el archivo existe
#         if not os.path.exists(nombre_archivo):
#             raise FileNotFoundError(f"El archivo {nombre_archivo} no se encuentra")
        
#         # Abir el archivo SQL
#         with open(nombre_archivo, 'r') as sql_file:
#             sql_as_string = sql_file.read()
        
#         # Ejecutar el SQL
#         cur.executescript(sql_as_string)
        
#     except Exception as e:
#         # Imprimir el error si lo hay
#         print("Error al ejecutar el archivo SQL:", str(e))



def ejecutar_sql(nombre_archivo, cur):
    try:
        archivo_path = Path(nombre_archivo)
        
        # Verificar si el archivo existe
        if not archivo_path.exists():
            raise FileNotFoundError(f"El archivo {archivo_path} no se encuentra")
        
        # Leer el archivo SQL
        with archivo_path.open('r') as sql_file:
            sql_as_string = sql_file.read()
        
        # Ejecutar el SQL
        cur.executescript(sql_as_string)
        
    except Exception as e:
        # Imprimir el error si lo hay
        print("Error al ejecutar el archivo SQL:", str(e))




# esta podemos usarla para limpiar las bases de datos, del semestre pasado
def identify_and_remove_outliers(conn, columns, threshold=2.1):
    # Leer datos desde la base de datos SQL
    df = pd.read_sql("SELECT * FROM all_employees", conn)

    for column in columns:
        Q1 = np.quantile(df[column], 0.25)
        Q3 = np.quantile(df[column], 0.75)
        IQR = Q3 - Q1
        upper = Q3 + threshold * IQR
        lower = Q1 - threshold * IQR

        # Usar SQL para eliminar outliers
        query = f"DELETE FROM all_employees WHERE {column} > {upper} OR {column} < {lower}"
        conn.execute(query)
        conn.commit()
        
        
#Se podria hacer una función para determinar la matriz de confusion
#Este lo tomo del trabajo del semestre pasado para ver si funciona.
def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix


#IMPUTADORES:
def imputar_f(df, list_cat):
    # Seleccionar las columnas categóricas
    df_c = df[list_cat]

    # Imputar valores faltantes para las columnas categóricas con el valor más frecuente
    imputer_c = SimpleImputer(strategy='most_frequent')
    X_c = imputer_c.fit_transform(df_c)
    df_c = pd.DataFrame(X_c, columns=df_c.columns)

    # Concatenar los DataFrames nuevamente
    df = pd.concat([df_c], axis=1)
    
    return df


def sel_variables(modelos,X,y,threshold):
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos



def preparar_datos (df):
   #######Cargar y procesar nuevos datos ######
   #### Cargar modelo y listas 
   
    list_cat=joblib.load("list_cat.pkl")
    list_dummies=joblib.load("list_dummies.pkl")
    var_names=joblib.load("var_names.pkl")
    scaler=joblib.load( "scaler.pkl") 

    ####Ejecutar funciones de transformaciones
    
    df=imputar_f(df,list_cat)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['perf_2023','EmpID2'])]
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    X=X[var_names]
    return X

    #####imputar datos numericos
def impute_columns(df, columns, strategy): #Función que imputa datos para variables numéricas
  imputer = SimpleImputer(strategy=strategy)
  for column in columns:
    column_imputed = imputer.fit_transform(df[column].values.reshape(-1, 1))
    df[column] = column_imputed.flatten()
  return df

def ct(columns):
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [columns])], remainder='passthrough')
  X = np.array(ct.fit_transform(X))
  return ct


# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k):
  rfe = RFE(model, n_features_to_select=k, step=1)
  fit = rfe.fit(X, y)
  X_new = fit.support_
  print("Num Features: %s" % (fit.n_features_))
  print("Selected Features: %s" % (fit.support_))
  print("Feature Ranking: %s" % (fit.ranking_))
  return X_new
