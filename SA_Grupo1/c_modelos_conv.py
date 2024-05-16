import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf #Para kas redes neuronales
from sklearn import metrics ### para analizar modelo
import pandas as pd

####instalar paquete !pip install keras-tuner
import keras_tuner as kt
from keras.utils import to_categorical
from tensorflow.keras.metrics import AUC


### cargar bases_procesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

x_train[0]

############################################################
################ Preprocesamiento ##############
############################################################

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train.max()
x_train.min()


x_train /=254 ### escalarlo para que quede entre 0 y 1, con base en el valor máximo
x_test /=254

###### verificar tamaños

x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)



##########################################################
################ Redes convolucionales ###################
##########################################################




cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])


y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Train the model for 10 epochs
cnn_model.fit(x_train, y_train_encoded, batch_size=100, epochs=10, validation_data=(x_test, y_test_encoded))


cnn_model.summary()

#######probar una red con regulzarización L2
reg_strength = 0.001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.3  


cnn_model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 2 outputs para 2 clases
])

#Complilamos el modelo con categorical_crossentropy y ADAM ya que tenemos 4 clases
cnn_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


# Train the model for 3 epochs
cnn_model2.fit(x_train, y_train_encoded, batch_size=100, epochs=3, validation_data=(x_test, y_test_encoded))


cnn_model2.summary()



#####################################################
###### afinar hiperparameter ########################
#####################################################



##### función con definicion de hiperparámetros a afinar
hp = kt.HyperParameters()

def build_model(hp):
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.05)
    reg_strength = hp.Float("reg_strength", min_value=0.0001, max_value=0.0005, step=0.0001)
    # No need to tune optimizer in this context
    # optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']) 
    

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 output neurons for 4 classes
    ])
    
  
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=[AUC(name='auc')]  # Add AUC metric with name 'auc'
    )
    return model



tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_auc',
    max_trials=10,
    directory='my_dir',
    project_name='alzheimers_classification'
)

tuner.search(x_train, y_train_encoded, epochs=3, validation_data=(x_test, y_test_encoded)) 

best_model = tuner.get_best_models(num_models=1)[0]

tuner.results_summary()
best_model.summary()


#################### Mejor redes ##############
best_model.save('/salidas/best_alzheimers_model.h5')
loaded_model = tf.keras.models.load_model('/salidas/best_alzheimers_model.h5')
loaded_model.summary()
