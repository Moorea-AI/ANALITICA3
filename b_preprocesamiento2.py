################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                       ALEJANDRA AGUIRRE                      #
#                    AURA LUZ MORENO - MOOREA                  #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

def ejecutar_sql (nombre_archivo, cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close
  cur.executescript(sql_as_string)
  
  #### Cargar paquetes
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import sys ## saber ruta de la que carga paquetes

#se generan las url para conectar las bases de datos desde git hub
employee = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/employee_survey_data.csv'
general = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/general_data.csv'
manager = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/manager_survey.csv'
retirement = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/retirement_info.csv'

# se leen los archivos csv
df_employee=pd.read_csv(employee)
df_general=pd.read_csv(general)
df_manager=pd.read_csv(manager)
df_retirement=pd.read_csv(retirement)


# resumen de información de las tablas
df_employee.info()
df_general.info()
df_manager.info()
df_retirement.info()


#### Convertir campos a formato fecha
df_employee["DateSurvey"]=pd.to_datetime(df_employee['DateSurvey'], format="%Y/%m/%d")
df_general["InfoDate"]=pd.to_datetime(df_general['InfoDate'], format="%Y/%m/%d")
df_manager["SurveyDate"]=pd.to_datetime(df_manager['SurveyDate'], format="%Y/%m/%d")
df_retirement["retirementDate"]=pd.to_datetime(df_retirement['retirementDate'], format="%Y/%m/%d")

## eliminar columnas
df_employee =df_employee.drop(["Unnamed: 0"], axis=1)
df_general =df_general.drop(["Unnamed: 0"], axis=1)
df_manager =df_manager.drop(["Unnamed: 0"], axis=1)
df_retirement =df_retirement.drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)

### por facilidad se convierten los nombres de las columnas a minuscula
df_employee = df_employee.rename(columns=lambda x: x.lower())
df_general = df_general.rename(columns=lambda x: x.lower())
df_manager = df_manager.rename(columns=lambda x: x.lower())
df_retirement = df_retirement.rename(columns=lambda x: x.lower())

df_employee.head(5)

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


