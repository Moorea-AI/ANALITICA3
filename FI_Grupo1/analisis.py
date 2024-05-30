import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure


# Cargo los datos históricos y las predicciones
df_historico = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv').dropna()
df_resultados = pd.read_csv('grupo_1.csv')


print(df_resultados.head())

# Revisemos la estad´sitica descriptiva: media, mediana, max, min, etc
descriptive_stats = df_resultados['int_rc'].describe()
print(descriptive_stats)

median_int_rc = df_resultados['int_rc'].median()
iqr_int_rc = df_resultados['int_rc'].quantile(0.75) - df_resultados['int_rc'].quantile(0.25)

print(f"Mediana: {median_int_rc}")
print(f"IQR (Rango Intercuartílico): {iqr_int_rc}")

# Cómo está distribuido int_rc?
plt.figure(figsize=(10, 6))
sns.histplot(df_resultados['int_rc'], kde=True, bins=30)
plt.title('Distribución de int_rc')
plt.xlabel('int_rc')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot para identificar si tengo valores atipicos en int_rc
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_resultados['int_rc'])
plt.title('Boxplot de int_rc')
plt.xlabel('int_rc')
plt.show()

# Identificar valores atípicos
Q1 = df_resultados['int_rc'].quantile(0.25)
Q3 = df_resultados['int_rc'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_resultados[(df_resultados['int_rc'] < lower_bound) | (df_resultados['int_rc'] > upper_bound)]
print(f"Número de valores atípicos: {len(outliers)}")
print("Valores atípicos:")
print(outliers)

outliers_stats = outliers['int_rc'].describe()
print("Estadísticas de los valores atípicos:")
print(outliers_stats)







# Nuevamente retomo los encoders para los datos históricos
categorical_features = ['HomeOwnership', 'Education', 'MaritalStatus']
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df_historico[feature] = le.fit_transform(df_historico[feature])

# Selecciono las variables relevantes
features = ['CreditScore', 'DebtRatio', 'Assets', 'Age', 'NumberOfDependents',
            'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome', 'NumberOfTimesPastDue',
            'EmploymentLength', 'HomeOwnership', 'Education', 'MaritalStatus',
            'YearsAtCurrentAddress']
target = 'NoPaidPerc'

X_historico = df_historico[features]
y_historico = df_historico[target]

# Y se hace el escalamiento tipico
scaler = StandardScaler()
X_historico = scaler.fit_transform(X_historico)

# Partimos los datos nuevamente en train y test
X_train, X_test, y_train, y_test = train_test_split(X_historico, y_historico, test_size=0.2, random_state=42)

# Aplico gradient boosting en los datos históricos
best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# y hacemos loas predicciones nuevamente para anlizar el error
y_pred_test = best_model.predict(X_test)

# Calculamos la predicción del error
errors = y_test - y_pred_test

# y se ingresa en un dataframe
df_errors = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test,
    'Error': errors
})

# Resumen de las estadisticas de los errores
error_summary = df_errors['Error'].describe()
print(error_summary)

# Distribución de los eerrores
plt.figure(figsize=(10, 6))
sns.histplot(df_errors['Error'], kde=True, bins=30)
plt.title('Distribución de los errores de predicción')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()

# Distribución delos atipicos
plt.figure(figsize=(10, 6))
sns.boxplot(df_errors['Error'])
plt.title('Boxplot de los errores de predicción')
plt.xlabel('Error')
plt.show()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(df_errors['Actual'], df_errors['Predicted'])
plt.plot([min(df_errors['Actual']), max(df_errors['Actual'])], [min(df_errors['Actual']), max(df_errors['Actual'])], color='red')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Métricas del error
mse = mean_squared_error(df_errors['Actual'], df_errors['Predicted'])
mae = mean_absolute_error(df_errors['Actual'], df_errors['Predicted'])
medae = median_absolute_error(df_errors['Actual'], df_errors['Predicted'])

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MedAE: {medae}')