import numpy as np
import joblib ### para cargar array
########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo
import pandas as pd
from sklearn import tree
import cv2 ### para leer imagenes jpeg
### pip install opencv-python
from matplotlib import pyplot as plt #

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical


### cargar bases_procesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')



############################################################
################ Preprocesamiento ##############
############################################################

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255

###### verificar tamaños

x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

##### convertir a 1 d array ############
x_train2 = x_train.reshape(5121, 100*100*3)  # 5121 images in x_train
x_test2 = x_test.reshape(1279, 100*100*3)  # 1279 images in x_test
x_train2.shape
x_test2.shape

rl = LogisticRegression()
rlmodel = rl.fit(x_train2, y_train)
predrltrain = rlmodel.predict(x_train2)
print(metrics.classification_report(y_train, predrltrain))

predrltest=rlmodel.predict(x_test2)
print(metrics.classification_report(y_test, predrltest))


############Analisis problema ###########
#### me interesa recall: de los enfermos que los pueda detectar, sin embargo
#### el problema es que puede generar mucho trabajo porque clasifica a 
####la mayoria como con neumonía, entonces usaremos el AUC que mide la capacidad e clasificación de neumoinía en balance con los noramles mal calsificados 





############################################################
################ Probar modelos de tradicionales#########
############################################################
rf=RandomForestClassifier()
rf.fit(x_train2, y_train)

preddtctrain=rf.predict(x_train2)
print(metrics.classification_report(y_train, preddtctrain))

preddtctest=rf.predict(x_test2)
print(metrics.classification_report(y_test, preddtctest))





############################################################
################ Probar modelos de redes neuronales #########
############################################################
y_train1=to_categorical(y_train)
y_test1=to_categorical(y_test)

fc_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]), #Se toma x_train original y no el x2, convierte las tres dimensiones en una sola dimensión
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

##### configura el optimizador y la función para optimizar ##############

fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])


#####Entrenar el modelo usando el optimizador y arquitectura definidas #########
fc_model.fit(x_train, y_train1,  epochs=20, validation_data=(x_test, y_test1))
#batch_size=100,

#########Evaluar el modelo ####################
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test1, verbose=2)  
print("Test recall:", test_recall)
fc_model.predict(x_test)
x_test.shape

pred_test=(fc_model.predict(x_test)>0.80).astype("int")
pred_test.shape

pred_test1=np.argmax(pred_test, axis=1)
y_test2=np.argmax(y_test1, axis=1)

cm = metrics.confusion_matrix(y_test2, pred_test1)  # Create confusion matrix
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'])
disp.plot()
plt.show()


print(metrics.classification_report(y_test1, pred_test))



#################### exportar red ##############
# guardar modelo

fc_model.save('path_to_my_model.h5') 

