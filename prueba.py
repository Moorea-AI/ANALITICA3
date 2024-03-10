# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from itertools import product
import sys
import sqlite3 as sql
import a_funciones as funciones  # Asegúrate de importar la función o ajustar según sea necesario
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, f1_score



# Conexión a la base de datos
conn = sql.connect("db_empleados")
df = pd.read_sql("SELECT * FROM all_employees", conn)
df.drop('index', axis=1, inplace=True)

# 1. Eliminación de columnas no relevantes
columns_to_drop = ['resignationReason', 'retirementDate', 'retirementType', 'InfoDate', 'DateSurvey', 'SurveyDate']
df.drop(columns=columns_to_drop, inplace=True)

# 2. Codificación One-Hot
columns_to_encode = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'EducationField',
                     'BusinessTravel', 'Department', 'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                     'PerformanceRating', 'StockOptionLevel', 'WorkLifeBalance']
df = pd.get_dummies(df, columns=columns_to_encode, dtype=int)

# 3. Normalización de columnas numéricas
scaler = StandardScaler()
ScalerList = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
              'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
              'YearsWithCurrManager']
df[ScalerList] = scaler.fit_transform(df[ScalerList])

# 4. Selección de Variables - Método KBest
XKbest = df.drop("Attrition", axis=1)
yKbest = df["Attrition"]
k_best = SelectKBest(score_func=f_classif, k=16)
X_best = k_best.fit_transform(XKbest, yKbest)
selected_featuresKbest = XKbest.columns[k_best.get_support()]
print("Características seleccionadas por KBest:")
print(selected_featuresKbest)

# 5. Selección de Variables - Método LASSO
X_lasso = df.drop("Attrition", axis=1)
y_lasso = df["Attrition"]
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_lasso, y_lasso)
lasso_coefficients = pd.DataFrame({'Feature': X_lasso.columns, 'Coefficient': lasso_model.coef_})
selected_features_lasso = lasso_coefficients[lasso_coefficients['Coefficient'] != 0]['Feature']
print("Características seleccionadas por Lasso:")
print(selected_features_lasso)

# 6. Selección de Variables - Método Wrapper
X_wrapper = df.drop("Attrition", axis=1)
y_wrapper = df["Attrition"]
model = LogisticRegression(max_iter=10000)
scaler = StandardScaler()
X_wrapper_std = scaler.fit_transform(X_wrapper)
X_new_selected = funciones.recursive_feature_selection(X_wrapper_std, y_wrapper, model, 16)
selected_features_df = pd.DataFrame(X_wrapper_std[:, X_new_selected], columns=X_wrapper.columns[X_new_selected])
print(selected_features_df)

# 7. Modelo de Regresión Logística
X_train, X_test, y_train, y_test = train_test_split(df[selected_features_df.columns], y_wrapper, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f'Exactitud en el conjunto de prueba: {accuracy:.3f}\n')
print('Matriz de Confusión:')
print(conf_matrix)
print('\nReporte de Clasificación:')
print(classification_rep)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
f1 = f1_score(y_test, y_pred)


# Precisión: 0.663
# Sensibilidad (Recall): 0.161
# Especificidad: 0.984
# Puntaje F1: 0.259


print(f'Precisión: {precision:.3f}')
print(f'Sensibilidad (Recall): {recall:.3f}')
print(f'Especificidad: {specificity:.3f}')
print(f'Puntaje F1: {f1:.3f}')

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.show()









