import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar los datos históricos
url_historico = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
df_historico = pd.read_csv(url_historico)

# Cargar los datos de los nuevos clientes
url_nuevos = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
df_nuevos = pd.read_csv(url_nuevos)

# Preprocesamiento (convertir variables categóricas a numéricas)
categorical_features = ['HomeOwnership', 'Education', 'MaritalStatus']
for feature in categorical_features:
    df_historico[feature] = pd.factorize(df_historico[feature])[0]
    df_nuevos[feature] = pd.factorize(df_nuevos[feature])[0]

# Seleccionar las variables relevantes
features = ['CreditScore', 'DebtRatio', 'Assets', 'Age', 'NumberOfDependents',
            'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome', 'NumberOfTimesPastDue',
            'EmploymentLength', 'HomeOwnership', 'Education', 'MaritalStatus',
            'YearsAtCurrentAddress']
target = 'NoPaidPerc'

X_historico = df_historico[features]
y_historico = df_historico[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_historico, y_historico, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

# Predecir la probabilidad de no pago para los nuevos clientes
X_nuevos = df_nuevos[features]
y_pred_nuevos = model.predict(X_nuevos)

# Asignar la tasa de interés (individual)
df_nuevos['int_rc'] = y_pred_nuevos * 100 

# Guardar los resultados en el formato solicitado
df_resultados = df_nuevos[['ID', 'int_rc']]
df_resultados.to_csv('grupo_1-LinearRegression.csv', index=False)

print("Archivo 'grupo_1-LinearRegression.csv' generado correctamente.")

