import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Cargar datos
url_historicos = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
url_nuevos = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'

df_historicos = pd.read_csv(url_historicos)
df_nuevos = pd.read_csv(url_nuevos)

# Preprocesamiento de datos (simplificado)
# Convertir categóricas a numéricas y manejar valores faltantes
df_historicos = pd.get_dummies(df_historicos).fillna(0)
df_nuevos = pd.get_dummies(df_nuevos).fillna(0)

# Asegurarse de que ambos DataFrames tengan las mismas columnas
missing_cols = set(df_historicos.columns) - set(df_nuevos.columns)
for col in missing_cols:
    df_nuevos[col] = 0

# Asegurarse de que los DataFrames tengan las mismas columnas (excepto 'NoPaidPerc' en df_nuevos)
df_nuevos = df_nuevos[df_historicos.columns.drop('NoPaidPerc')]

# Dividir los datos históricos en características (X) y target (y)
X = df_historicos.drop(columns=['ID', 'NoPaidPerc'])
y = df_historicos['NoPaidPerc']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

# Predecir la probabilidad de no pago para nuevos clientes
X_nuevos = df_nuevos.drop(columns=['ID'])
df_nuevos['NoPaidPerc_Pred'] = model.predict(X_nuevos)

# Definir la tasa de interés asociada al riesgo
def calcular_tasa_riesgo(prob_no_pago):
    # Fórmula simplificada, ajustar según sea necesario
    return min(0.2, max(0.01, prob_no_pago))

df_nuevos['int_rc'] = df_nuevos['NoPaidPerc_Pred'].apply(calcular_tasa_riesgo)

# Seleccionar columnas requeridas para la entrega
df_entrega = df_nuevos[['ID', 'int_rc']]
df_entrega.to_csv('grupo_1-RandomForest.csv', index=False)
