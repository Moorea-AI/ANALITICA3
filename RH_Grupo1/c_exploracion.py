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
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import RH_Grupo1.a_funciones as a_funciones ### archivo de funciones propias
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import graphviz
from sklearn.pipeline import Pipeline





# Conexión a la base de datos db_empleados
conn = sql.connect("db_empleados")
cur = conn.cursor()  # para ejecutar consultas SQL en la base de datos


# Cargar datos desde SQLm seleccionamos todo de la tabla all_employees
df = pd.read_sql("select * from all_employees", conn)

df.columns


################################################################
#                                                              #
#   ANÁLISIS DE LAS VARIABLES DE LA BD_EMPLEADOS EN FIRME      #              #
#                                                              #    
################################################################

# EnvironmentSatisfaction: 1. Low, 2. Medium, 3. High, 4. Very High
# JobSatisfaction:  1. Low, 2. Medium, 3. High, 4. Very High
# WorkLifeBalance: 1. Bad, 2. Good, 3. Better, 4. Best
# Age: Integer
# BusinessTravel: 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 
# Department: Sales, Research & Development, Human Resources
# DistanceFromHome: distancia en kilómetros
# Education: 1. Below college, 2. College, 3. Bachelor, 4. Master, 5. Doctor
# EducationField: Life Sciences, Other, Medical, Marketing, Technical Degree, Human Resources
# Gender: Male or Female
# JobLevel: Job level at company on a scale of 1 to 5
# JobRole: Healthcare Representative, Research Scientist, Sales Executive, Human Resources, Research Director, Laboratory Technician, Manufacturing Director,Sales Representative, Manager
# MaritalStatus: Married, Single, Divorced
# MonthlyIncome': Monthly income in rupees per month (Rupees? es de la India?)
# NumCompaniesWorked : Total number of companies the employee has worked for
# PercentSalaryHike: Percent salary hike for last year
# StockOptionLevel: Stock option level of the employee  
# TotalWorkingYears:  Total number of years the employee has worked so far
# TrainingTimesLastYear: Number of times training was conducted for this employee last year
# YearsAtCompany: Total number of years spent at the company by the employee
# YearsSinceLastPromotion: Number of years since last promotion
# YearsWithCurrManager: Number of years under current manager
# InfoDate: Date when gneral information about employees was extracted
# JobInvolvement: 1. Low, 2. Medium, 3. High, 4. Very High
# PerformanceRating: 1. Low, 2: Good, 3. Excellent, 4. Outstanding
# SurveyDate: Date in which surveys (manager and satisfaction) were carried out
# Attrition: Yes. No. Es un término que se utiliza en recursos humanos y gestión empresarial para referirse a la tasa de rotación o la tasa de desgaste de empleados en una organización. Representa la proporción de empleados que dejan la empresa en un período de tiempo determinado, ya sea debido a renuncias, jubilaciones, despidos u otras razones.
# retirementDate: Date employee quit the company
# retirementType:  None, 'Resignation', 'Fired'. Fired when decision was made for the company and resignation when it was made for the employee
# resignationReason: None, 'Others', 'Stress', 'Salary'


################################################################
#                                                              #
#        DISTRIBUCIÓN DE GÉNERO                                #
#       Es una empresa mayoritariamente masculina?             # 
#                                                              #
#       Si, la empresa tiene un 60% hombres y un 40% mujeres   #
#                                                              #
################################################################



#Esto se nos ocurrió el 09/03/2024 y decidimos agregarlo
# Gráfico de torta para la distribución del género
gender_counts = df['Gender'].value_counts()
labels = gender_counts.index
sizes = gender_counts.values

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Distribución de Género')
plt.show()
      
################################################################
#                                                              #
#        1. ANÁLISIS DE RESIGNATION REASON.                    #
#       ¿Porqué renuncian y con qué frecuencia?                # 
#                                                              #
#       Se observa que "otros" es la variable más frecuente    #
#       Por ende se descarta el salario y el estrés como       #
#       la razón fundamental                                   #                   
#                                                              #
################################################################

df['resignationReason'].unique()
      
#Asignamos la columna a una variable
df['resignationReason'].replace('nan', np.nan, inplace=True)
reason_counts = df['resignationReason'].dropna().value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=reason_counts.index, y=reason_counts.values, palette="viridis")
plt.title('Frecuencia de Razones de Renuncia')
plt.xlabel('Razón de Renuncia')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.show()
#Podemos ver que "otras" son las razones más allá del salario o el estrés


################################################################
#                                                              #
#        2. ANÁLISIS DE LA EDAD (AGE).                         #
#       ¿Cómo está distribuida la edad en la empresa?          # 
#                                                              #
#       Como lo hablamos en el preprocesamiento, queremos      #
#       explorar ¿cuál es la edad que tiene más deserción      #
#       serán los jóvenes o los más mayores?                   $
#                                                              #                                                              #
#       Podemos observar que la mayoría del personal actual    #
#       se encuentra entre 28 y 48 años                        #                   
#                                                              #
################################################################
    
# Visualización de la distribución de la edad, con un rango entre 28 y 48 años
sns.histplot(df['Age'], bins=30, kde=False, color=plt.cm.viridis(0.3), alpha=0.7)
plt.title('Distribución de edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
note_text = "Se puede ver la distribución entre 25 y 35 años"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()


######################################################################
#                                                                    #
#   3. ANÁLISIS DE LA EDAD (AGE) VS TASA DE ABANDONO (Attrition)     #
#   ¿Son los más jóvenes los que abandonan? los más viejos?          # 
#                                                                    #
#   En este caso es 0 para los que aún permanecen en la empresa y    #
#   1 para los que renunciaron.                                      #
#   Se puede ver que si hay una tendencia a que las personas más     #
#   jóvenes se retiren de la empresa ya que su rango de dad para     #                                                               #
#   la tasa de renuncia está más abajo en el boxplot.                #
#                                                                    #
######################################################################
 
plt.figure(figsize=(12, 8))
sns.boxplot(x='Attrition', y='Age', data=df, palette="Set2")
plt.title('Relación entre Edad y Attrition')
plt.xlabel('Attrition o rotación')
plt.ylabel('Edad')
plt.show()


######################################################################
#                                                                    #
#   4. Pero... Cómo está la satisfacción laboral?                    #             
#                                                                    #  
#   ANÁLISIS DE JOB SATISFACTION (satisfacción laboral)              #
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   Podemos observar que la mayor frecuencia está en 3 y 4 que es    #
#   "High" (alta) y "Very High" (muy alta), es decir que los         #
#   empleados están renunciando por otra razón que debemos           #
#   explorar.                                                        #
#                                                                    #
######################################################################
 
 
#Distribución de la satisfacción laboral
plt.figure(figsize=(8, 6))
sns.countplot(x='JobSatisfaction', data=df, color=plt.cm.viridis(0.3))
plt.title('Distribución de Satisfacción Laboral')
plt.xlabel('Satisfacción Laboral')
plt.ylabel('Frecuencia')
note_text = "1. Bajo, 2. Medio, 3. Alto, 4. Muy alto"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()


######################################################################
#                                                                    #
#   5. Y qué pasa si la comparamos con la variable objetivo?         #             
#                                                                    #  
#   ANÁLISIS DE JOB SATISFACTION (satisfacción laboral) VS ATTRITION # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   En este caso es cero para los que continuan en la emrpesa y 1    #
#   para los que renunciaron y ya no están. Obtenemos lo esperado    #
#   Las personas que ya no están tenían menos satisfacción laboral   #
#   que las que continúan. Sin embargo vemos un balance entre ambos  #
#   boxplot que nos podría indicar que esta no es una variable       #
#   concluyente                                                      #
#                                                                    #
######################################################################
 
# Abandono versus la satisfaccion laboral
# Pero como se relaciona la tasa de abandono con la satisfacción laboral?
scaler = MinMaxScaler()
df['JobSatisfaction'] = df['JobSatisfaction'].astype(float)
df[['JobSatisfaction', 'Attrition']] = scaler.fit_transform(df[['JobSatisfaction', 'Attrition']])

sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title('Relación entre satisfacción laboral y retención del empleado')
plt.xlabel('Attrition / Abandono / Desgaste')
plt.ylabel('Satisfacción laboral')
note_text = "A mayor satisfacción laboral, menos tasa de abandono"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()
# En este caso es cero para "no" y  uno para "si", podemos ver lo esperado, entre más satisfacción laboral más retención.


######################################################################
#                                                                    #
#   6. Cuáles son los departamentos con más renuncias?               #             
#                                                                    #  
#   ANÁLISIS DE DEPARTMENT VS ATTRITION                              # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   En el caso de Sales y Research vemos comportamientos muy         #
#   similares. Esto no ocurre con RRHH quienes tanto los que         #
#   se retiraron como los que se quedaron tienen una alta            #
#   satisfacción laboral. Es decir que estos empleados tienen        #
#   otras razones para renunciar que se deberian explorar en         #
#   otro trabajo y presentrlo en las recomendaciones finales         #
#                                                                    #
######################################################################


plt.figure(figsize=(12, 8))
sns.boxplot(x='Department', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Departamento y Abandono')
plt.xlabel('Departamento')
plt.ylabel('Satisfacción Laboral')
plt.show()

######################################################################
#                                                                    #
#   7. Qué pasa con el nivel educativo?
#                                                                    #  
#   ANÁLISIS DE Education VS ATTRITION                               # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   Qué pasa con los que tienen doctorado? son los que menos         #
#   están, aunque también no son los que tengan mayor número         #
#   de empleados.Pero si tienen la mayor tasa de abandono            #
#   Los cargos con estudios más bajos tienen menos tasa de           #
#   abandono. Esto se puede explicar con los campos de acción.       #  
#                                                                    #
######################################################################


# Relación entre la satisfacción laboral y la educación
#Las personas con menor nivel educativo y menos contentas en su trabajo abandonan más que las que si?
plt.figure(figsize=(12, 8))
sns.boxplot(x='Education', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Educación y Abandono')
plt.xlabel('Nivel de Educación')
plt.ylabel('Satisfacción Laboral')
note_text = "Education: 1. Below college, 2. College, 3. Bachelor, 4. Master, 5. Doctor"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)

plt.show()

######################################################################
#                                                                    #
#   8. Qué pasa con el nivel laboral?
#                                                                    #  
#   ANÁLISIS DE JobLevel VS ATTRITION                                # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   Este es un factor común para todos. El nivel del cargo           #
#   medio y el más alto son los que presentan menos inconfomidades   #
#   que los dos primeros.                                            #                                                                    #
#                                                                    #
######################################################################

# Relación entre la satisfacción laboral y el nivel de cargo
# Qué relación hay entre los niveles directivos y la satisfacción laboral?
# Y que pasa son los niveles más bajos?
plt.figure(figsize=(12, 8))
sns.boxplot(x='JobLevel', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Nivel de Cargo y Abandono')
plt.xlabel('Nivel de Cargo')
plt.ylabel('Satisfacción Laboral')
note_text = "Se mide en una escala de 1 a 5"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)

plt.show()


######################################################################
#                                                                    #
#   9. Y el estado civil? se van más los solteros, casados?          #
#                                                                    #  
#   ANÁLISIS DE MaritalStatus VS ATTRITION                           # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   Tenemos a los solteros con mayor deserción, seguido de los       #
#   casados. Los dirvorciados son los que menos se retiran de la     #
#   compañia.                                                        #                                                                    #
#                                                                    #
######################################################################

# Relación entre la satisfacción laboral y el estado civil
# Quienes renuncia más? casados? solteros? divorciados?
plt.figure(figsize=(12, 8))
sns.boxplot(x='MaritalStatus', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Estado Civil y Abandono')
plt.xlabel('Estado Civil')
plt.ylabel('Satisfacción Laboral')
plt.show()

#La complementaremos con esta:

plt.figure(figsize=(10, 6))
sns.countplot(x='MaritalStatus', hue='Attrition', data=df, palette='viridis')
plt.title('Relación entre Estado Civil y Abandono')
plt.xlabel('Estado Civil')
plt.ylabel('Cantidad')
plt.show()


######################################################################
#                                                                    #
#   10. Y los que se demoran más en llegar al trabajo?               #                                                                    #  
#                                                                    #
#   ANÁLISIS DE DistanceFromHome VS ATTRITION                        # 
#   Tiene relación con Attrition? Veamos como está distribuida.      #
#                                                                    #
#   Podemos observar que están muy balanceadas por lo que la         #
#   distancia no es un factor que defina la variable                 #
#                                                                    #
######################################################################

# Relación entre la satisfacción laboral y la distancia al trabajo
# Quiénes renuncian más? los que viven más lejos? los que viven más cerca?
plt.figure(figsize=(12, 8))
sns.boxplot(x='Attrition', y='DistanceFromHome', data=df)
plt.title('Relación entre Distancia al Trabajo y Abandono Laboral')
plt.xlabel('Abandono Laboral')
plt.ylabel('Distancia al Trabajo')
plt.show()

###############################
######ARCHIVO PROFE
###############################
###############################


#### Exploración de todas las variables numéricas con histograma.
df.columns
df_numeric = df.select_dtypes(include=['float64', 'int64']).drop(['index', 'EmployeeID'], axis=1)

fig = df_numeric.hist(bins=50, figsize=(40, 30), grid=False, ec='black')
plt.show()


####explorar relación con variable respuesta #######

######################################################################
#                                                                    #
#   ANALISIS DE CORRELACION DE LAS VARIABLES NUMÉRICAS               #                                                                    #  
#                                                                    #
#   Observaremos como estan correlacionadas positiva o negativamente #
#   todas las variables del dataframe                                #
#                                                                    #
######################################################################


###Analizar correlación de numéricas con un rango amplio,las de rangopequeño se pueden analizar como categóricas
df.info()


df_numeric = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_numeric.corr()
attrition_correlation = corr_matrix["Attrition"].sort_values(ascending=False)
attrition_correlation

# CORRELACIONES POSITIVAS
# NumCompaniesWorked         0.041503
# PercentSalaryHike          0.032533
# PerformanceRating          0.023403

# Esto quiere decir que los empleados que han tenido otros 
# trabajos, los que han tenido mayores aumentos de salario y
# mejor desempeño, son los que tienden a abandonar más rápido,
# aunque no es una correlación muy alta.

# CORRELACIONES CASI CERO
# EmployeeID                -0.004729
# StockOptionLevel          -0.006839
# DistanceFromHome          -0.009730

# Estas no tienen un impacto muy determinante en el estudio.

# CORRELACIONES NEGATIVAS
# JobLevel                  -0.010290
# Education                 -0.015111
# JobInvolvement            -0.015588
# MonthlyIncome             -0.031176
# YearsSinceLastPromotion   -0.033019
# TrainingTimesLastYear     -0.049431
# WorkLifeBalance           -0.062975
# EnvironmentSatisfaction   -0.101625
# JobSatisfaction           -0.103068
# YearsAtCompany            -0.134392
# YearsWithCurrManager      -0.156199
# Age                       -0.159205
# TotalWorkingYears         -0.171050
    
# Los que están más satisfechos con el clima laboral, con su trabajo
# y que llevan más años con la compañia y con su jefe actual son más propensos
# a quedarse en la empresa a medida que pasan los años y se incrementa su edad


# Prepararemos el teerreno para un arbol de decisión:
# Se eliminan variables nulas y se seleccionan las variables numericas
df_arbol = df.select_dtypes(include=['float64', 'int64']).dropna()

# Separamos las variables predictoras de la variable objetivo
X = df_arbol.drop("Attrition", axis=1)   # X contendrá todas las variables excepto "Attrition"
y = df_arbol["Attrition"] # y será la variable que estamos tratando de predecir (Attrition)


# Entrenamos un arbol dedecision para hacer predicciones sobre nuevas instancias de datos
# Creamos un modelo de árbol de decisión con una profundidad máxima de 3.
model = DecisionTreeClassifier(max_depth=3) #Limitamos el arbol a 3 para evitar el sobreajuste
model.fit(X, y) # Entrenamos el modelo con las variables predictoras (X) y la variable objetivo (y).

# Obtenemos la importancia de cada variable según el árbol de decisión.
importances = model.feature_importances_ #importancia de las columnas
feature_names = X.columns  #nombre de todas las columnas

# Organizamos la importancia de todas las variables en un DataFrame para una mejor visualización.
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

#Imprimimos la importancia de las variables según el árbol de decisión, ordenadas de mayor a menor importancia.
print("Importancia de las variables según el árbol de decisión:\n", feature_importance_df.sort_values(by="Importance", ascending=False))


# Importancia de las variables según el árbol de decisión:
#                      Feature  Importance
# 13        TotalWorkingYears    0.540448
# 10       NumCompaniesWorked    0.121857
# 2   EnvironmentSatisfaction    0.110722
# 5                       Age    0.097991
# 9             MonthlyIncome    0.089455
# 18           JobInvolvement    0.039527
# 12         StockOptionLevel    0.000000
# 17     YearsWithCurrManager    0.000000
# 16  YearsSinceLastPromotion    0.000000
# 15           YearsAtCompany    0.000000
# 14    TrainingTimesLastYear    0.000000
# 0                     index    0.000000
# 11        PercentSalaryHike    0.000000
# 1                EmployeeID    0.000000
# 8                  JobLevel    0.000000
# 7                 Education    0.000000
# 6          DistanceFromHome    0.000000
# 4           WorkLifeBalance    0.000000
# 3           JobSatisfaction    0.000000
# 19        PerformanceRating    0.000000


# TotalWorkinYears Total de años trabajados
# Es la variable más importante según el árbol de decisión. Indica que el número total de años trabajados por un 
# empleado tiene un impacto significativo en la predicción de si abandonará la empresa o no.


# NumCompaniesWorked o Numero de compañias donde ha trabajado antes: La segunda variable más importante. Sugiere que 
# la cantidad de empresas para las que ha trabajado un empleado también es un factor clave para prever si abandonará 
# la empresa.


# EnvironmentSatisfaction o Satisfacción con el entorno de trabajo: Esta variable también tiene un impacto considerable. 
# Indica que la satisfacción del empleado con el entorno de trabajo es un factor importante

 
# Age o Edad: La edad del empleado también es un predictor relevante para determinar la probabilidad de abandono.

# MonthlyIncome o Ingreso mensual: El ingreso mensual del empleado también se considera en la predicción.



# AHORA QUE CONOCEMOS LAS VARIABLES MÁS IMPORTANTES, EXPLOREMOS ESPECÍFICMENTE ESAS CON LA VARIABLE OBJETIVO

# TotalWorkingYears
sns.boxplot(x='Attrition', y='TotalWorkingYears', data=df)
plt.title('Relación entre la variable objetivo Attrition y TotalWorkingYears')
plt.xlabel('Abandono')
plt.ylabel('Total de años trabajados')
plt.show()


#NumCompaniesWorked
sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=df)
plt.title('Relación entre la variable objetivo y NumCompaniesWorked')
plt.xlabel('NumCompaniesWorked')
plt.ylabel('Numero de empresas donde trabajó')
plt.show()


#Age
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title('Relación entre la variable objetivo y Age')
plt.xlabel('Age')
plt.ylabel('Edad')
plt.show()

#MonthlyIncome
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Relación entre la variable objetivo y MonthlyIncome')
plt.xlabel('MonthlyIncome')
plt.ylabel('Salario mensual')
plt.show()


