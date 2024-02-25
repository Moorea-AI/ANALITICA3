### Funciones

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Para poder ver gráficamente las variables
import seaborn as sns
import os  #Paquete OS: https://docs.python.org/es/3.10/library/os.html
from sklearn.metrics import confusion_matrix #Ya que es un problema de clasificación, quiero usar la matriz de confusion para predecir si un empleado se queda o no.


#Se toman del archivo del profe por si nos sirven de algo más adelante:
from sklearn.impute import SimpleImputer 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 


###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas
def ejecutar_sql(nombre_archivo, cur):
    try:
        # Mirar si el archivo existe
        if not os.path.exists(nombre_archivo):
            raise FileNotFoundError(f"El archivo {nombre_archivo} no se encuentra")
        
        # Abir el archivo SQL
        with open(nombre_archivo, 'r') as sql_file:
            sql_as_string = sql_file.read()
        
        # Ejecutar el SQL
        cur.executescript(sql_as_string)
        
    except Exception as e:
        # Imprimir el error si lo hay
        print("Error al ejecutar el archivo SQL:", str(e))
