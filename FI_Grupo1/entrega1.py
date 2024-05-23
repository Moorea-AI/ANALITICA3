import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos históricos
url_historico = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
df_historico = pd.read_csv(url_historico)

# Cargar los datos de los nuevos clientes
url_nuevos = 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
df_nuevos = pd.read_csv(url_nuevos)

# Mostrar las primeras filas de los datos
print(df_historico.head())
print(df_nuevos.head())

# Limpieza básica (por ejemplo, eliminar filas con valores nulos)
df_historico = df_historico.dropna()
df_nuevos = df_nuevos.dropna()

# Convertir variables categóricas a numéricas
categorical_features = ['HomeOwnership', 'Education', 'MaritalStatus']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df_historico[feature] = le.fit_transform(df_historico[feature])
    df_nuevos[feature] = le.transform(df_nuevos[feature])
    label_encoders[feature] = le

# Seleccionar las variables relevantes
features = ['CreditScore', 'DebtRatio', 'Assets', 'Age', 'NumberOfDependents',
            'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome', 'NumberOfTimesPastDue',
            'EmploymentLength', 'HomeOwnership', 'Education', 'MaritalStatus',
            'YearsAtCurrentAddress']
target = 'NoPaidPerc'

X_historico = df_historico[features]
y_historico = df_historico[target]
X_nuevos = df_nuevos[features]

# Escalar las variables numéricas
scaler = StandardScaler()
X_historico = scaler.fit_transform(X_historico)
X_nuevos = scaler.transform(X_nuevos)

# Análisis exploratorio de datos (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df_historico['NoPaidPerc'], kde=True, bins=30)
plt.title('Distribución de NoPaidPerc')
plt.show()

# Boxplot de las variables numéricas con respecto a la variable objetivo
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_historico[feature], y=df_historico['NoPaidPerc'])
    plt.title(f'Boxplot of {feature} vs NoPaidPerc')
    plt.show()

# Evaluar varios modelos
X_train, X_test, y_train, y_test = train_test_split(X_historico, y_historico, test_size=0.2, random_state=42)

# Definir los modelos
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Evaluar los modelos
results = {}
for model_name, model in models.items():
    if model_name == "Logistic Regression":
        # Binarizar la variable objetivo para la regresión logística
        y_train_bin = (y_train > 0.5).astype(int)
        y_test_bin = (y_test > 0.5).astype(int)
        model.fit(X_train, y_train_bin)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        mse = mean_squared_error(y_test, y_pred_prob)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_prob)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    print(f"{model_name} - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

# Mostrar los resultados
results_df = pd.DataFrame(results).T
print(results_df)

# Seleccionar el mejor modelo (en este ejemplo, supongamos que es Random Forest)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_historico, y_historico)

# Predecir la probabilidad de no pago para los nuevos clientes
y_pred_nuevos = best_model.predict(X_nuevos)

# Asignar la tasa de interés (individual)
df_nuevos['int_rc'] = y_pred_nuevos * 100

# Guardar los resultados en el formato solicitado
df_resultados = df_nuevos[['ID', 'int_rc']]
df_resultados.to_csv('grupo_1_best_model.csv', index=False)

print("Archivo 'grupo_1_best_model.csv' generado correctamente.")
