# Importamos librerias para el manejo del sistema
import sys
import os
import tensorflow as tf
# Importamos librerias de Keras desde Tensorflow
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator  # Para preprocesar las imagenes
from tensorflow.python.keras import optimizers  # Optimizadores para el algoritmo
from tensorflow.python.keras.models import Sequential  # Unir capas para redes neuronales sequenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation  # Para las capas de la NN
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D  # Para las capas convolucionales de la NN
from tensorflow.python.keras import backend as K  # Para sesiones previas de Keras

# Limpiamos cualquier sesión previa de Keras
K.clear_session()
tf.compat.v1.disable_eager_execution()

# Creamos variables globales para parametros futuros
'''Directorios de los datasets'''
trainData = './data/train'
testData = './data/test'
'''Parametros de la NN'''
epocas = 20
altura, longitud = 100, 100
batchSize = 32
pasos = 1000
pasosValidacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamanoFiltro1 = (3, 3)
tamanoFiltro2 = (2, 2)
tamanoPool = (2, 2)
clases = 3
lr = 0.0005

# Preprocesamos el training dataset
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

# Preprocesamos el validation dataset
validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Creamos la variable que alojara las imagenes del dataset
imagenTrain = entrenamiento_datagen.flow_from_directory(
    trainData,
    target_size=(altura, longitud),
    batch_size=batchSize,
    class_mode='categorical'
)

imagenTest = validation_datagen.flow_from_directory(
    testData,
    target_size=(altura, longitud),
    batch_size=batchSize,
    class_mode='categorical'
)

# Creamos la CNN
cnn=Sequential()

## Primera capa
cnn.add(Convolution2D(filtrosConv1, tamanoFiltro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamanoPool))
## Segunda capa
cnn.add(Convolution2D(filtrosConv2, tamanoFiltro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamanoPool))

# Comenzamos a clasificar
cnn.add(Flatten()) #La imagen la aplanamos a una capa
cnn.add(Dense(256, activation='relu')) #Conectamos todas las neuronas
cnn.add(Dropout(0.5)) #Apagamos el 50% de las neuronas para generalizacion
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

# Entrenamos el algoritmo
cnn.fit(imagenTrain, steps_per_epoch=pasos, epochs=epocas, validation_data=imagenTest, validation_steps=pasosValidacion)

# Guardamos el modelo para hacerlo offline
dir='./modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')