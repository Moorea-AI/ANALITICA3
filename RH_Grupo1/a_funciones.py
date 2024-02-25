### Funciones

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Para poder ver gráficamente las variables
import seaborn as sns

from sklearn.metrics import confusion_matrix #Ya que es un problema de clasificación, quiero usar la matriz de confusion para predecir si un empleado se queda o no.


#Se toman del archivo del profe por si nos sirven de algo más adelante:
from sklearn.impute import SimpleImputer 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 