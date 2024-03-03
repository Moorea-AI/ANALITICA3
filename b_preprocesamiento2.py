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

#### Cargar paquetes
import pandas as pd # para manejo de datos
import sqlite3 as sql ### para bases de datos sql
import a_funciones as a_funciones ## funciones creadas en el archivo de funciones

#Hacemos llamado a las bases de datos desde GitHub
df_employee = pd.read_csv('https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/employee_survey_data.csv')
df_general = pd.read_csv('https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/general_data.csv')
df_manager = pd.read_csv('https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/manager_survey.csv')
df_retirement = pd.read_csv('https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/retirement_info.csv')

# Verificamos la correcta visualización
df_employee.sort_values(by=['EmployeeID']).head(10)
df_general.sort_values(by=['EmployeeID']).head(10)
df_manager.sort_values(by=['EmployeeID']).head(10)
df_retirement.sort_values(by=['EmployeeID']).head(10)

# Verificamos el tipo de información que tiene cada uno
df_employee.info(verbose=True)
df_general.info()
df_manager.info()
df_retirement.info()

#Existen datos faltantes? Revisamos los atípicos:
print(df_employee.isnull().sum())
print(df_general.isnull().sum())
print(df_manager.isnull().sum()) #No tiene nulos
print(df_retirement.isnull().sum())

## Eliminamos columnas que no se necesitan ya que son índices que no tienen relación entre si
df_employee =df_employee.drop(["Unnamed: 0"], axis=1)
df_general =df_general.drop(["Unnamed: 0"], axis=1)
df_manager =df_manager.drop(["Unnamed: 0"], axis=1)
df_retirement =df_retirement.drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)


# Se construye una sola BD la cual pivotamos por el ID del empleado para que queden solo
# las filas que tienen los datos completos.
df = df_employee.merge(df_general, on='EmployeeID', how='inner')\
                        .merge(df_manager, on='EmployeeID', how='inner')\
                        .merge(df_retirement, on='EmployeeID', how='left')              


# Revisamos las filas que tienen los datos nulos
print(df.isnull().sum())

# Consultamos el diccionario de datos para determinar cuáles columnas son relevantes para elmodelo y cuales no
# Para proceder a eliminarlas
print(df.columns)


#Algunas columnas no aportan mucho al modelo, por ejemplo 
# EmployeeCount: Employee count managed 
# Over18: Whether the employee is above 18 years of age or not ya que tenemos una columna de edad que sirve para lo mismo
# StandardHours: Standard hours of work for the employee no es relevante yaq ue el numero de horas trabajadas ya se infiere

df.drop(columns=["EmployeeCount", "Over18", "StandardHours"],inplace=True)



#Revisamos los nulos columna por columna:

##############################################
#EnvironmentSatisfaction' tiene 200 nulos. Revisaremos la mediana y la media para determinar el valor más adecuado
df['EnvironmentSatisfaction'].value_counts()  #3 es el valor que más tiene
df['EnvironmentSatisfaction'].mean()  #2.7251
df['EnvironmentSatisfaction'].median() #3
#Como la mediana y la media se aproximan a 3, se rellenarán los nulos con 3
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].fillna(3)

##############################################
# 'JobSatisfaction', hacemos lo mismo que el anterior, con 160
df['JobSatisfaction'].value_counts() #4 y 3 son los valores que más se repiten
df['JobSatisfaction'].mean() #2.72 que se aproxima a 3
df['JobSatisfaction'].median() # Mediana de 3
# El valor que menos afecta la medición es 3, por lo que se rellenan con este.
df['JobSatisfaction'] = df['JobSatisfaction'].fillna(3)

##############################################
# 'WorkLifeBalance', hacemos lo mismo, tiene 304 filas nulas
df['WorkLifeBalance'].value_counts() #3 es el valor que más se repite
df['WorkLifeBalance'].mean() #2.76
df['WorkLifeBalance'].median() #3
# El valor que menos afecta la medición es 3, por lo que se rellenan con este.
df['WorkLifeBalance'] = df['WorkLifeBalance'].fillna(3)


##############################################
# 'NumCompaniesWorked', hacemos lo mismo, tiene 152 filas nulas
df['NumCompaniesWorked'].value_counts() #1 es el valor que más se repite
df['NumCompaniesWorked'].mean() #2.69
df['NumCompaniesWorked'].median() #2
#En este caso los cambiamos a CERO, ya que hay personas que pueden estar en su primer empleo y todavia no cumplen 1 año.
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(0)

##############################################
# 'TotalWorkingYears', hacemos lo mismo, tiene 72 filas nulas
df['TotalWorkingYears'].value_counts() #10 es el valor que más se repite
df['TotalWorkingYears'].mean() #11.27
df['TotalWorkingYears'].median() #10
#Para este caso, a pesr de que la media y la mediana estan en 11.27 y 10, lo cambiaremos por el total de años que lleva en la empresa 'Years at company'
df['TotalWorkingYears'].fillna(df['YearsAtCompany'], inplace=True)

##############################################
# 'retirementDate' tiene valores nulos, que es lógico ya que hay muchos empleados activos los cuales NO tienen fecha de retiro al momento.
# Estos valores nulos se quedan asi por ahora, sin embargo, hay que revisar el formato de fecha de este campo
df['retirementDate'].info() #Tiene datos tipo objeto
df['retirementDate'] = pd.to_datetime(df['retirementDate'], dayfirst=True)
# Queda con tipo fecha
print(df.dtypes)


##############################################
# 'Attrition' no esta en el diccionario de datos de la BD. Al traducir es un "desgaste", como una tasa de deserción,
#  la cual tiene una traducción dificil, aunque podríamos tomarlo como booleano, algo asi como renunció? SI__ NO __
# "Attrition" es un término que se utiliza en recursos humanos y gestión empresarial para referirse a la 
#  tasa de rotación o la tasa de desgaste de empleados en una organización. Representa la proporción de 
#  empleados que dejan la empresa en un período de tiempo determinado, ya sea debido a renuncias, 
#  jubilaciones, despidos u otras razones.
#
#   ****************** IMPORTANTE *******************
#
#                   VARIABLE ATTRITION  
# 
#   Esta es la candidata perfecta para ser la variable objetivo, 
#   ya que determina si un empleado se retiró o no
#
#   *************************************************
#
contenido_columna = df['Attrition']
print(contenido_columna)
# Rellenaremos los valores nulos con NO, ya que corresponde a empleados que aún están dentro de la compañia
df['Attrition'].fillna('No', inplace=True)
attrition_mapping = {'Yes': 1, 'No': 0}
df['Attrition'] = df['Attrition'].map(attrition_mapping)
df['Attrition'].unique()
df['Attrition'].value_counts() # Con esto asumimos que tenemos 29.592 empleados activos y 5.688 retirados
# Aunque me deja una duda: En el planteamiento del problema dice que es una empresa con 4.000 empleados y aqui 
#  aparecen 29.592. 



##############################################
# 'JobSatisfaction' de tipo float a  tipo Int 
df['JobSatisfaction'].dtype
df['JobSatisfaction'] = df['JobSatisfaction'].astype(float).astype(int)
df['JobSatisfaction'].unique()    



#Revisamos nuevamente la base de datos para verificar que no haya quedado ningun nulo sin motivo aparente
print(df.isnull().sum())
print(df.columns)


# Y ahora si la creamos en SQL
conn = sql.connect("db_empleados")  # creacion de la base de datos
cursor = conn.cursor() # para funcion execute
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
cursor.fetchall() # para ver en tablas las bases de datos

df.to_sql("all_employees", conn, if_exists="replace") #Y agregamos
# tenemos 35.280 datos entre los que renunciaron y los activos.-

#Y probaremso algunas consultas para verificar el funcionamiento correcto

# Nos da curiosidad saber si las personas más jóvenes son las que más se retiran:
pd.read_sql("""SELECT Age, COUNT(*) as Retirements 
                                    FROM all_employees 
                                    WHERE retirementDate IS NOT NULL 
                                    GROUP BY Age
                                    ORDER BY Retirements DESC""", conn)


#Se nos ocurrren más preguntas pero las contestaremos en la exploración


#En el archivo de funciones teniamos una función para detección de los atipicos:
a_funciones.identify_and_remove_outliers(conn, ['MonthlyIncome', 'TrainingTimesLastYear', 'YearsAtCompany', 'TotalWorkingYears'])

# CONVERTIR A STRING
# BusinessTravel de object a string
# Department de object a string
# EducationField de object a string
# Gender de object a string
# JobRole de object a string
# MaritalStatus de object a string
# retirementType
# resignationReason

columns_to_convert_str = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'retirementType', 'resignationReason'
]
# Y cambiamos a tipo string
for column in columns_to_convert_str:
    df[column] = df[column].astype(str)
print(df.dtypes)

# CONVERTIR A FECHA
# DateSurvey
# InfoDate 
columns_to_convert_datetime = [
    'DateSurvey', 'InfoDate', 'SurveyDate'
]
# tipo datetime
for column in columns_to_convert_datetime:
    df[column] = pd.to_datetime(df[column], errors='coerce')
print(df.dtypes)

df['DateSurvey'].unique() 
df['InfoDate'].unique() 
df['SurveyDate'].unique() 
    

#Vamos a crear un archivo .sql donde pondremos los P.A. para manipular los datos
# Y a tomar solamente la información relevante para el trabajo que nos compete, con
# los retirados hasta 2016 ya que estamos parados en el 01/01/2017

df.info()
df.describe(include='all')

df.to_sql("all_employees", conn, if_exists="replace")

df = pd.read_sql("SELECT * FROM all_employees", conn)

cur=conn.cursor()


a_funciones.ejecutar_sql('b_preprocesamiento.sql',cur)

# Al hacer una lectura notamos que tenemos datos duplicados 4 veces. Estos serán retirados
pd.read_sql("""SELECT * FROM retirados_2016""", conn)



# LA BASE DE DATOS QUEDA CON ESTAS COLUMNAS
# EnvironmentSatisfaction: 1. Low, 2. Medium, 3. High, 4. Very High
# JobSatisfaction:  1. Low, 2. Medium, 3. High, 4. Very High
# WorkLifeBalance: 1. Bad, 2. Good, 3. Better, 4. Best
# Age: Integer
# BusinessTravel: 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 
# Department: Sales, Research & Development, Human Resources
# DistanceFromHome: distancia en kilometros
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
