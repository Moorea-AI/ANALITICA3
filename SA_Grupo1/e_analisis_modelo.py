import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

x_train = joblib.load('salidas/x_train.pkl')
y_train = joblib.load('salidas/y_train.pkl')
x_test = joblib.load('salidas/x_test.pkl')
y_test = joblib.load('salidas/y_test.pkl')

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=254 
x_test /=254

##### cargar modelo  ######

model=tf.keras.models.load_model('salidas/best_alzheimers_model.h5')

####desempeño en evaluación para grupo 1 (tienen Alzheimer o no) #######
prob=model.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en test")### conocer el comportamiento de las probabilidades para revisar threshold

threshold_alz=0.52

#pred_test=(modelo.predict(x_test)>=threshold_alz).astype('int')
pred_test = np.argmax(model.predict(x_test), axis=1)

print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demented', 'NonDemented'])
disp.plot()

### desempeño en entrenamiento #####
prob=model.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en train")### conocer el comportamiento de las probabilidades para revisar threshold

pred_proba = model.predict(x_train)
pred_proba_demented = pred_proba[:, 1]

pred_train = (pred_proba_demented >= threshold_alz).astype('int')
#pred_train=(prob>=threshold_alz).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demented', 'NonDemented'])
disp.plot()

####desempeño en evaluación para grupo 1 (No tienen Alzheimer) #######
########### ##############################################################

prob=model.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en test")### conocer el comportamiento de las probabilidades para revisar threshold

threshold_no_alz=0.50

pred_proba = model.predict(x_test)
pred_proba_demented = np.sum(pred_proba[:, [0, 1, 3]], axis=1)  # Combine probabilities for dementia classes

pred_test = (pred_proba_demented >= threshold_no_alz).astype('int')  # Threshold based on combined probability

#pred_test=(model.predict(x_test)>=threshold_no_alz).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demented', 'NonDemented'])
disp.plot()

### desempeño en entrenamiento #####
prob=model.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("Probabilidades imágenes en train")### conocer el comportamiento de las probabilidades para revisar threshold

pred_proba = model.predict(x_train)
pred_proba_demented = np.sum(pred_proba[:, [0, 1, 3]], axis=1)  # Combine probabilities for dementia classes
pred_train = (pred_proba_demented >= threshold_no_alz).astype('int')  # Threshold based on combined probability

#pred_train=(prob>=threshold_no_alz).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demented', 'NonDemented'])
disp.plot()

####### clasificación final ################

# prob=model.predict(x_test)

# clas=['alz' if prob >0.52 else 'No alz' if prob <0.50 else "No ident" for prob in prob]

# clases, count =np.unique(clas, return_counts=True)

# count*100/np.sum(count)

prob = model.predict(x_test)

clas = []
for p in prob:
    if p[0] > 0.52:  # Comparación de la probabilidad de la clase "alz"
        clas.append('Alz')
    elif p[0] < 0.50:  # Comparación de la probabilidad de la clase "No alz"
        clas.append('No Alz')
    else:
        clas.append('No Ident')

clases, count = np.unique(clas, return_counts=True)

porcentaje = count * 100 / np.sum(count)

print("Porcentaje de cada clase:")
for c, p in zip(clases, porcentaje):
    print(f"{c}: {p:.2f}%")
