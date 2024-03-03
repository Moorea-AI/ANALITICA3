#### Cargar paquetes
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import sys ## saber ruta de la que carga paquetes


## conectamos las funciones
import a_funciones as funciones

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

### por facilidad se convierten los nombres de las columnas a minuscula
df_employee = df_employee.rename(columns=lambda x: x.lower())
df_general = df_general.rename(columns=lambda x: x.lower())
df_manager = df_manager.rename(columns=lambda x: x.lower())
df_retirement = df_retirement.rename(columns=lambda x: x.lower())

print(df_employee.shape)#se consulta el tamaño del DF
print(df_general.shape)#se consulta el tamaño del DF
print(df_manager.shape)#se consulta el tamaño del DF
print(df_retirement.shape)#se consulta el tamaño del DF

# resumen de información de las tablas
df_employee.info()
df_general.info()
df_manager.info()
df_retirement.info()


 ###consultar el número de nulos
print(df_employee.isnull().sum())
print(df_general.isnull().sum())
print(df_manager.isnull().sum())
print(df_retirement.isnull().sum())



#Las variables que tienen datos nulos son:

#environmentsatisfaction 50
#jobsatisfaction 40
#worklifebalance 76
#numcompaniesworked 38
#totalworkingyears 18
#resignationreason 70


#### Convertir campos a formato fecha
df_employee["datesurvey"]=pd.to_datetime(df_employee['datesurvey'], format="%Y/%m/%d")
df_general["infodate"]=pd.to_datetime(df_general['infodate'], format="%Y/%m/%d")
df_manager["surveydate"]=pd.to_datetime(df_manager['surveydate'], format="%Y/%m/%d")
df_retirement["retirementdate"]=pd.to_datetime(df_retirement['retirementdate'], format="%Y/%m/%d")


## eliminar columnas
df_employee =df_employee.drop(["unnamed: 0"], axis=1)
df_general =df_general.drop(["unnamed: 0","over18", "employeecount","standardhours"], axis=1)
df_manager =df_manager.drop(["unnamed: 0"], axis=1)
df_retirement =df_retirement.drop(["unnamed: 0","unnamed: 0.1"], axis=1)

df_employee.head(5)

df_manager.head(5)

#crear base de datos en SQL
conn= sql.connect("data\\db_empleados") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.
cur=conn.cursor() ### ejecutar funciones  en BD


### Llevar tablas a base de datos
df_employee.to_sql("employee",conn,if_exists="replace")
df_general.to_sql("general",conn,if_exists="replace")
df_manager.to_sql("manager",conn,if_exists="replace")
df_retirement.to_sql("retirement",conn,if_exists="replace")

cur.execute("Select name from sqlite_master where type='table'") ### consultar bases de datos
cur.fetchall()

BD = pd.read_sql("""
SELECT
    e.employeeid,
    e.environmentsatisfaction,
    e.jobsatisfaction,
    e.worklifebalance,
    e.datesurvey,
    m.jobinvolvement,
    m.performancerating,
    m.surveydate,
    r.attrition,
    r.retirementdate,
    r.retirementtype,
    r.resignationreason,
    g.age,
    g.businesstravel,
    g.department,
    g.distancefromhome,
    g.education,
    g.educationfield,
    g.gender,
    g.jobLevel,
    g.jobrole,
    g.maritalStatus,
    g.monthlyIncome,
    g.numcompaniesworked,
    g.percentsalaryhike,
    g.stockoptionlevel,
    g.totalworkingyears,
    g.trainingtimeslastyear,
    g.yearsatcompany,
    g.yearssincelastpromotion,
    g.infodate
FROM employee e
INNER JOIN manager m ON e.employeeid = m.employeeid
INNER JOIN retirement r ON e.employeeid = r.employeeid
INNER JOIN general g ON e.employeeid = g.employeeid""", conn)

BD.to_sql("BD",conn,if_exists="replace")

BD

pd.read_sql("""select employeeid,count(*)
                            from BD
                            group by employeeid""", conn)

pd.read_sql("""select datesurvey,count(*)
                            from employee
                            group by datesurvey""", conn)

pd.read_sql("""select surveydate,count(*)
                            from manager
                            group by surveydate""", conn)

pd.read_sql("""SELECT strftime('%Y',retirementdate) as fecha,
                                count(*) as cnt
                                from BD
                                group by fecha""", conn)

pd.read_sql("""select strftime('%Y', retirementdate) AS FECHA,count(*)
                            from retirement
                            group by FECHA""", conn)

pd.read_sql("""select strftime('%Y', infodate) AS FECHA,count(*)
                            from general
                            group by FECHA""", conn)

##### 1. Filtrar datos por empleados que tengan evaluación de desempeño en el último año 2016
pd.read_sql("""SELECT strftime('%Y', surveydate) AS fecha,
       jobinvolvement AS cnt
        FROM BD
        WHERE strftime('%Y', surveydate) = '2016'
        GROUP BY fecha""", conn)

##Imputación de datos

##as variables que tienen datos nulos son:

#environmentsatisfaction 50
#jobsatisfaction 40
#worklifebalance 76
#numcompaniesworked 38
#totalworkingyears 18
#resignationreason 70

## Se usa la función unique para verificar los datos nulos
print("environmentsatisfaction: ")
print(BD.environmentsatisfaction.unique())
print(" ")
print("jobsatisfaction: ")
print(BD.jobsatisfaction.unique())
print(" ")
print("numcompaniesworked: ")
print(BD.numcompaniesworked.unique())
print(" ")
print("totalworkingyears: ")
print(BD.totalworkingyears.unique())
print(" ")
print("resignationreason: ")
print(BD.resignationreason.unique())
print(" ")
print("worklifebalance: ")
print(BD.worklifebalance.unique())



## verificamos que tipo de variable es
BD.info()

### Con la función creada tratamos los datos nulos de la base de detos que son de tipo númerico
BD = funciones.impute_columns(df = BD, columns = ['environmentsatisfaction', 'jobsatisfaction', 'worklifebalance', 'numcompaniesworked', 'totalworkingyears'], strategy = 'median')

### para la variable categorica resignationreason tratamos los nulos diciendo que se encuentran activos debido a que posiblemente esa sea la razón por la cual hay un dato nulo
BD['resignationreason'] = BD['resignationreason'].fillna('Activo')


