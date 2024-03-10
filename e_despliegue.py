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

import a_funciones as funciones
import pandas as pd
import sqlite3 as sql
import joblib
import openpyxl
import numpy as np

if __name__ == "__main__":
    # Conectarse a la base de datos
    conn = sql.connect("db_empleados")
    cur = conn.cursor()

    # Ejecutar SQL de preprocesamiento inicial
    funciones.ejecutar_sql('preprocesamientos2.sql', cur)
    df = pd.read_sql('''select * from all_employees''', conn)

    # Otras transformaciones en Python (imputación, dummies y selección de variables)
    df_t = funciones.preparar_datos(df)

    # Cargar modelo y predecir
    m_lreg = joblib.load("salidas\\m_lreg.pkl")
    predicciones = m_lreg.predict(df_t)
    pd_pred = pd.DataFrame(predicciones, columns=['pred_perf_2024'])

    # Crear base con predicciones
    perf_pred = pd.concat([df['EmpID2'], df_t, pd_pred], axis=1)

    # Llevar a BD para despliegue
    perf_pred.loc[:, ['EmpID2', 'pred_perf_2024']].to_sql("perf_pred", conn, if_exists="replace")

    # Ver predicciones bajas
    emp_pred_bajo = perf_pred.sort_values(by=["pred_perf_2024"], ascending=True).head(10)
    emp_pred_bajo.set_index('EmpID2', inplace=True)
    pred = emp_pred_bajo.T

    # Agregar coeficientes y exportar a Excel
    coeficientes = pd.DataFrame(np.append(m_lreg.intercept_, m_lreg.coef_), columns=['coeficientes'])
    pred.to_excel("salidas\\prediccion.xlsx")
    coeficientes.to_excel("salidas\\coeficientes.xlsx")



