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

# Conexión a la base de datos
conn = sql.connect("db_empleados")
cur = conn.cursor()  # para ejecutar consultas SQL en la base de datos


# Cargar datos desde SQL
df = pd.read_sql("select * from all_employees", conn)

df.columns

df['resignationReason'].unique()

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

      

# Como lo hablamos en el preprocesamiento, queremos explorar cuál es el grupo de edad
#que tiene más deserción. Para esto, primero queremos ver dentro de la muestra la distribución 
# de la edad
# Visualización de la distribución de la edad
sns.histplot(df['Age'], bins=30, kde=False, color=plt.cm.viridis(0.3), alpha=0.7)
plt.title('Distribución de edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
note_text = "Se puede ver la distribución entre 25 y 35 años"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()


# El clima laboral y el indice de satisfacción con el trabajo
sns.histplot(df['JobSatisfaction'], bins=30, kde=False, color=plt.cm.viridis(0.3), alpha=1)
plt.title('Distribución de satisfacción laboral')
plt.xlabel('Satisfacción laboral')
plt.ylabel('Frecuencia')
note_text = "1. Low, 2. Medium, 3. High, 4. Very High"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()



# Abandono versus la satisfaccion laboral
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
plt.title('Relación entre satisfacción laboral y retención del empleado')
plt.xlabel('Attrition / Abandono / Desgaste')
plt.ylabel('Satisfacción laboral')
note_text = "A mayor satisfacción laboral, menos tasa de abandono"
plt.text(0.5, -0.2, note_text, ha='center', va='center', fontsize=10, color='gray', transform=plt.gca().transAxes)
plt.show()















###############################
######ARCHIVO PROFE
###############################
###############################

### explorar variable respuesta ###
fig=df.perf_2023.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

boxprops = dict(linestyle='-', color='black')
medianprops = dict(linestyle='-',  color='black')
fig=df.boxplot("perf_2023",patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=dict(color='black'),
                showmeans=True)
fig.grid(False)
plt.show()

####explorar variables numéricas con histograma
fig=df.hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show()


####explorar relación con variable respuesta #######

###Analizar correlación de numéricas con un rango amplio,las de rangopequeño se pueden analizar como categóricas
df.info()
continuas = ['perf_2023',
             'dias_lst_mov',
             'antiguedad_dias',
             'edad_dias',
             'PayRate2',
             'EmpSatisfaction',
             'EngagementSurvey'
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