"""
--------------------------------------------------------
TRABAJO FINAL DE ANALITICA 3 - UNIVERSIDAD DE ANTIOQUIA
--------------------------------------------------------
Por: Aura Luz Moreno Díaz
Fecha: 30/05/2024

Descripción:
Este código carga datos históricos y de nuevos clientes, realiza limpieza y preprocesamiento de datos, 
entrena y evalúa varios modelos de machine learning, selecciona el mejor modelo y predice la probabilidad de no pago 
para los nuevos clientes. Finalmente, guarda los resultados y proporciona un análisis de los resultados obtenidos.

Modelos evaluados:
- Regresión Lineal
- Bosques Aleatorios
- Árboles de decisión
- Gradient Boosting
- Support Vector Machine - SVM
- K-Nearest Regression
- Ridge
- Lasso

Métricas de evaluación:
- MSE (Error Cuadrático Medio)
- RMSE (Raíz del Error Cuadrático Medio)
- R2 (Coeficiente de Determinación)
- MAE (Error Absoluto Medio)
- MedAE (Mediana del Error Absoluto)
- Explained Variance (Varianza Explicada)

-------------------------------------------------------
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from matplotlib.pyplot import figure


# Carga de los datos históricos
url_historico = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
df_historico = pd.read_csv(url_historico)

# Carga de los datos de los nuevos clientes
url_nuevos = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
df_nuevos = pd.read_csv(url_nuevos)

# Muestra las primeras filas de los datos
print(df_historico.head())
print(df_nuevos.head())

# Limpieza básica (elimina filas con valores nulos)
df_historico = df_historico.dropna()
df_nuevos = df_nuevos.dropna()

# Convierte variables categóricas a numéricas
categorical_features = ['HomeOwnership', 'Education', 'MaritalStatus']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df_historico[feature] = le.fit_transform(df_historico[feature])
    df_nuevos[feature] = le.transform(df_nuevos[feature])
    label_encoders[feature] = le

# Seleccion de las variables relevantes
features = ['CreditScore', 'DebtRatio', 'Assets', 'Age', 'NumberOfDependents',
            'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome', 'NumberOfTimesPastDue',
            'EmploymentLength', 'HomeOwnership', 'Education', 'MaritalStatus',
            'YearsAtCurrentAddress']
target = 'NoPaidPerc'

X_historico = df_historico[features]
y_historico = df_historico[target]
X_nuevos = df_nuevos[features]

# Escala las variables numéricas
scaler = StandardScaler()
X_historico = scaler.fit_transform(X_historico)
X_nuevos = scaler.transform(X_nuevos)

# Análisis exploratorio de datos (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df_historico['NoPaidPerc'], kde=True, bins=30)
plt.title('Distribución de NoPaidPerc')
plt.show()

# Boxplot de las variables numéricas con respecto a la variable objetivo
# Se hace con un for para explorar todas las variables 
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_historico[feature], y=df_historico['NoPaidPerc'])
    plt.title(f'Boxplot of {feature} vs NoPaidPerc')
    plt.show()

sns.histplot(data=df_historico, x="NoPaidPerc")

df_hist_num = df_historico.select_dtypes(include=['number'])
figure(figsize=(20,6))
sns.heatmap(df_hist_num.corr(),cmap = sns.cubehelix_palette(as_cmap=True), annot = True, fmt = ".2f")


# Evaluar varios modelos
# 20% para los datos de prueba y 80% para entrenamiento.
X_train, X_test, y_train, y_test = train_test_split(X_historico, y_historico, test_size=0.2, random_state=42)

# Definimos los modelos
models = {
    "Linear Regression": LinearRegression(), # Es un modelo sencillo y rápido de entrenar, ideal para obtener una línea base y entender relaciones lineales entre las características y la variable objetivo.
    "Ridge Regression": Ridge(), # Modelos de regularización para manejar el problema de multicolinealidad y reducir el riesgo de sobreajuste en la regresión lineal
    "Lasso Regression": Lasso(), # Modelos de regularización para manejar el problema de multicolinealidad y reducir el riesgo de sobreajuste en la regresión lineal
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42), #Es un método de ensamble basado en múltiples árboles de decisión, que mejora la precisión y reduce el sobreajuste.
    "Decision Tree": DecisionTreeRegressor(random_state=42), #Es fácil de interpretar y visualizar, proporciona una buena base para entender cómo funcionan los métodos de ensamble como Random Forest y Gradient Boosting.
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42), # Es un método de ensamble potente que optimiza los errores residuales de los modelos previos, proporcionando alta precisión.
    "Support Vector Machines": SVR(), # Es un modelo versátil que puede manejar tanto problemas de clasificación como de regresión
    "K-Nearest Neighbors": KNeighborsRegressor() # Predice el valor de la variable objetivo para una observación basada en el promedio de los valores de las k observaciones más cercanas
}

# Evaluar los modelos
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse, # Valores más bajos indican un mejor ajuste del modelo a los datos.
        "RMSE": rmse, # Valores más bajos indican un mejor ajuste del modelo a los datos.
        "R2": r2, # Valores más cercanos a 1 indican un mejor ajuste del modelo a los datos.
        "MAE": mae, # Valores más bajos indican un mejor rendimiento del modelo.
        "MedAE": medae,
        "Explained Variance": evs  #Valores más cercanos a 1 indican un mejor ajuste del modelo a los datos.

    })

results_df = pd.DataFrame(results)
print(results_df)

# Análisis de los resultados
analysis_text = """
Análisis de los Resultados:

- R² (R cuadrado): Gradient Boosting tiene el R² más alto (0.7404), lo que significa que explica la mayor varianza en la variable objetivo (NoPaidPerc) en comparación con los otros modelos. Un R² más cercano a 1 indica un mejor ajuste.
- MSE (Error Cuadrático Medio): Gradient Boosting tiene el MSE más bajo (0.001170), lo que indica que sus predicciones están más cerca de los valores reales.
- RMSE (Raíz del Error Cuadrático Medio): Similar al MSE, un RMSE más bajo (0.034395) es mejor, y Gradient Boosting tiene el valor más bajo.
- MAE (Error Absoluto Medio): Aunque no es tan sensible a valores atípicos como el MSE, un MAE más bajo (0.022818) sigue siendo deseable, y Gradient Boosting lo tiene.
- MedAE (Mediana del Error Absoluto): Gradient Boosting tiene la MedAE más baja (0.016348), lo que sugiere que la mitad de sus predicciones tienen errores menores o iguales a este valor.

Los modelos con mejor rendimiento son: Gradient Boosting y Random Forest. Elegiremos Gradient Boosting.
Los modelos con rendimineto moderado son: KNN, regresión lineal y ridge.
Los modelos con bajo rendimiento son: SVM
El modelo con un rendimineto raro es el LASSO ya que tiene un R² negativo
"""
print(analysis_text)

# Mostrar los resultados
results_df = pd.DataFrame(results).T
print(results_df)


# best_model_name = results_df.loc[results_df['R2'].idxmax()]['Model']
# best_model = models[best_model_name]

print("El mejor modelo basado en los resultados proporcionados es el Gradient Boosting.")
print("Tiene el MSE más bajo (0.001170) y el R2 más alto (0.7404) entre todos los modelos evaluados.")
print("Esto indica que tiene la capacidad más sólida para explicar la variabilidad en la variable objetivo y hacer predicciones precisas.")


# El mejor modelo basado en los resultados proporcionados es el Gradient Boosting. Tiene el MSE más bajo y el R2 más alto entre todos los modelos evaluados. Esto indica que tiene la capacidad más sólida para explicar la variabilidad en la variable objetivo y hacer predicciones precisas.
best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_historico, y_historico)


X_nuevos = df_nuevos[features]

# Predecimos la probabilidad de no pago para los nuevos clientes
y_pred_nuevos = best_model.predict(X_nuevos)

#Asignamos la tasa de interés
df_nuevos['int_rc'] = y_pred_nuevos * 100

#Guardamos los resultados
df_resultados = df_nuevos[['ID', 'int_rc']]

# Los exportamos al archivo
df_resultados.to_csv('grupo_1.csv', index=False)


# Esto es solo para comprobar que los resultados estén en el formato correcto
df_grupo_1 = pd.read_csv("grupo_1.csv")
print(df_grupo_1.head(10))
df_grupo_1.shape


