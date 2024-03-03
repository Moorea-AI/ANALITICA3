################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                       ALEJANDRA AGUIRRE                      #
#                    AURA LUZ MORENO - MOOREA                  #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as a_funciones ### archivo de funciones propias
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Conexión a la base de datos db_empleados
conn = sql.connect("db_empleados")
cur = conn.cursor()  # para ejecutar consultas SQL en la base de datos


# Cargar datos desde SQLm seleccionamos todo de la tabla all_employees
df = pd.read_sql("select * from all_employees", conn)

df.columns

df['resignationReason'].unique()

################################################################
#                                                              #
#          ANALISIS DE LAS VARIABLES DE LA BD EN FIRME         #              #
#                                                              #    
################################################################

# EnvironmentSatisfaction: 1. Low, 2. Medium, 3. High, 4. Very High
# JobSatisfaction:  1. Low, 2. Medium, 3. High, 4. Very High
# WorkLifeBalance: Bad, Good, Better, Best
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
      
#Asignamos la columna a una variable
reason_counts = df['resignationReason'].value_counts()

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



# Relación entre la satisfacción laboral y el departamento
# Existen departamentos con mayor deserción?
# Cómo varia la deserción según el departamento?
plt.figure(figsize=(12, 8))
sns.boxplot(x='Department', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Departamento y Abandono')
plt.xlabel('Departamento')
plt.ylabel('Satisfacción Laboral')
plt.show()

# Relación entre la satisfacción laboral y la educación
#Las personas con menor nivel educativo y menos contentas en su trabajo abandonan más que las que si?
plt.figure(figsize=(12, 8))
sns.boxplot(x='Education', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Educación y Abandono')
plt.xlabel('Nivel de Educación')
plt.ylabel('Satisfacción Laboral')
plt.show()


# Relación entre la satisfacción laboral y el nivel de cargo
# Qué relación hay entre los niveles directivos y la satisfacción laboral?
# Y que pasa son los niveles más bajos?
plt.figure(figsize=(12, 8))
sns.boxplot(x='JobLevel', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Nivel de Cargo y Abandono')
plt.xlabel('Nivel de Cargo')
plt.ylabel('Satisfacción Laboral')
plt.show()


# Relación entre la satisfacción laboral y el estado civil
# Quienes renuncia más? casados? solteros? divorciados?
plt.figure(figsize=(12, 8))
sns.boxplot(x='MaritalStatus', y='JobSatisfaction', hue='Attrition', data=df)
plt.title('Relación entre Satisfacción Laboral, Estado Civil y Abandono')
plt.xlabel('Estado Civil')
plt.ylabel('Satisfacción Laboral')
plt.show()


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
fig=df.hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show()


####explorar relación con variable respuesta #######

###Analizar correlación de numéricas con un rango amplio,las de rangopequeño se pueden analizar como categóricas
df.info()
continuas = ['Age',
             'DistanceFromHome',
             'MonthlyIncome',
             'PercentSalaryHike',
             'StockOptionLevel',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'YearsAtCompany',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager'
             ]
scatter_matrix(df[continuas], figsize=(12, 8))
plt.show()
cont=df[continuas]
corr_matrix = cont.corr()
corr_matrix["perf_2023"].sort_values(ascending=False)

df.plot(kind="scatter",y="perf_2023",x="avg_perf")
plt.show()

##### analizar relación con categóricas ####

df.boxplot("perf_2023","GenderID",figsize=(5,5),grid=False)
df.boxplot("perf_2023","EmpSatisfaction",figsize=(5,5),grid=False)

#### también se pueden usar modelos para exploracion ######
####Ajustar un modelo para ver importancia de variables categóricas

####Crear variables para entrenar modelo
y=df.perf_2023
X=df.loc[:, ~df.columns.isin(['perf_2023','cnt_total','cnt_mov10','cnt_mov30','cnt_mov90','cnt_mov91','EmpID2'])]
X.info()
X_dummy=pd.get_dummies(X,columns=['DepID','level2','MaritalDesc','FromDiversityJobFairID','position','State','CitizenDesc','HispanicLatino','RaceDesc','RecruitmentSource'])
X_dummy.info()

#entrenar modelo
rtree=tree.DecisionTreeRegressor(max_depth=3)
rtree=rtree.fit(X=X_dummy,y=y)

####Analizar resultados del modelo
r = export_text(rtree,feature_names=X_dummy.columns.tolist(),show_weights=True)
print(r)
plt.figure(figsize=(40,40))
tree.plot_tree(rtree,fontsize=9,impurity=False,filled=True)
plt.show()

#####HAcer lista de variables importantes
d={"columna":X_dummy.columns,"importancia": rtree.feature_importances_}
df_import=pd.DataFrame(d)
pd.set_option('display.max_rows', 100)
df_import.sort_values(by=['importancia'],ascending=0)

##### ver graficamente las categorias más importantes

df.plot(kind="scatter",y="perf_2023",x="dias_lst_mov")

df.boxplot("perf_2023","MaritalDesc",figsize=(5,5),grid=False)
df.boxplot("perf_2023","DepID",figsize=(5,5))