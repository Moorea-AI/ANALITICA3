################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                       ALEJANDRA AGUIRRE                      #
#                    AURA LUZ MORENO - MOOREA                  #
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

## Eliminamos columnas que no se necesitan
df_employee =df_employee.drop(["Unnamed: 0"], axis=1)
df_general =df_general.drop(["Unnamed: 0"], axis=1)
df_manager =df_manager.drop(["Unnamed: 0"], axis=1)
df_retirement =df_retirement.drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)


# Se construye una sola BD
df = df_employee.merge(df_general, on='EmployeeID', how='inner')\
                        .merge(df_manager, on='EmployeeID', how='inner')\
                        .merge(df_retirement, on='EmployeeID', how='left')              


print(df.isnull().sum())
print(df.columns)

#Revisamos los nulos columna por columna:

##############################################
#EnvironmentSatisfaction' tiene 200 nulos. Revisaremos la mediana y la media para determinar el valor más adecuado
df['EnvironmentSatisfaction'].value_counts()  #3 es el valor que más tiene
df['EnvironmentSatisfaction'].mean()  #2.7236
df['EnvironmentSatisfaction'].median() #3
#Como la mediana y la media se aproximan a 3, se rellenarán los nulos con 3
df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].fillna(3)

##############################################
# 'JobSatisfaction', hacemos lo mismo que el anterior, con 160
df['JobSatisfaction'].value_counts() #4 y 3 es el valor que mas se repite
df['JobSatisfaction'].mean() #2.7272 que se aproxima a 3
df['JobSatisfaction'].median() # Mediana de 3
df['JobSatisfaction'] = df['JobSatisfaction'].fillna(3)





       'WorkLifeBalance', 'DateSurvey', 'Age', 'BusinessTravel', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'InfoDate', 'JobInvolvement', 'PerformanceRating', 'SurveyDate',
       'Attrition', 'retirementDate', 'retirementType', 'resignationReason'

#crear base de datos en SQL
conn= sql.connect("databases/db_empleados.sql") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.
cur=conn.cursor() ### ejecutar funciones  en BD

### Llevar tablas a base de datos
df_employee.to_sql("employee",conn,if_exists="replace")
df_general.to_sql("general",conn,if_exists="replace")
df_manager.to_sql("manager",conn,if_exists="replace")
df_retirement.to_sql("retirement",conn,if_exists="replace")


cur.execute("Select name from sqlite_master where type='table'") ### consultar bases de datos
cur.fetchall()


pd.read_sql("""select employeeid,count(*)
                            from employee
                            group by employeeid""", conn)


