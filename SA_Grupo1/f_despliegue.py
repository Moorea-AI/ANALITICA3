import numpy as np
import pandas as pd
import cv2
import funciones as fn
import tensorflow as tf
import openpyxl

import sys
sys.executable
sys.path

if __name__=="__main__":

    # Cargar datos
    path = '/data/despliegue/'
    files, x, _ = fn.img2data2(path)  # Cargar datos de despliegue

    x = np.array(x)  # Imágenes a predecir
    x = x.astype('float') / 255  # Convertir para escalar

    # Eliminar extensión a nombre de archivo
    files2 = [name.rsplit('.', 1)[0] for name in files]

    # Cargar modelo
    model = tf.keras.models.load_model('/salidas/best_alzheimers_model.h5')

    # Realizar predicciones
    prob = model.predict(x)

    # Clasificar las predicciones
    clas = []
    for p in prob:
        if p > 0.508:
            clas.append('Alz')
        elif p < 0.5015:
            clas.append('No Alz')
        else:
            clas.append('No Ident')

    # Crear DataFrame con resultados
    res_dict = {
        "paciente": files2,
        "clas": clas
    }
    resultados = pd.DataFrame(res_dict)

    # Guardar resultados en un archivo Excel
    resultados.to_excel('/salidas/clasificados.xlsx', index=False)
