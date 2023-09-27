# ------------------ Elementos de Deep Learning ------------------ #
# ------------------ Librarias ------------------ #

import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import json
import random
import pickle

import warnings
warnings.filterwarnings('ignore')

# ------------------ Desarrollo ------------------ #
# ------------------ Usando TensorFlow y Keras------------------ #
## Ejemplo: Entrenando una red neuronal sobre el conjunto de datos Iris ##

# Import pyplot
import matplotlib.pyplot as plt

# Dummy from sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

'''
# Cargando el conjunto de datos iris
iris = load_iris()

# print(iris)

# Valores correspondientes a las características de las flores
# print(pd.DataFrame(iris.data, columns=iris.feature_names).sample(10))

# Tipo de flor correspondiente (Versicolor = 0, Virginica = 1, Setosa = 2)
# print(pd.DataFrame(iris.target, columns=['type']).sample(10))

# Separando los datos en conjuntos de entrenamiento en prueba

# Divide el conjunto de datos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1234)

# Api de Keras para definir el modelo

# Crea un modelo secuencial de TensorFlow
model = Sequential([
    Dense(10, activation = 'sigmoid', input_shape=(4,)),
    Dense(3, activation = 'softmax')
])

# Compila el modelo
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'sgd',
    metrics = ['accuracy']
)

# Entrena el modelo
history = model.fit(
    X_train,
    y_train,
    epochs = 50,
    batch_size = 5,
    verbose = 1
)

# Evualuando el perfomance del modelo

# Evalua el modelo en el conjunto de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Imprime la precisión del modelo
print('Precisión del modelo:', accuracy)

# Graficando el historial de entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'loss'], loc='upper left')
plt.show()
'''
## ------------------ Un modelo de Deep Learning para generar ------------------ ##
### ------------------ Entrenando un modelo de Deep Learning para generar un ChatBot de reconocimiento de palabras clave 🤖 ## ------------------ ###
### Conjunto de datos sample.json ###

# Carga los datos
with open('sample.json') as f:
    conversations = json.load(f)

# Imprimimos el contenido
from pprint import pprint
# pprint(conversations)

# Crea un diccionario con las respuestas del
# bot por categoria y lo salvamos

category_answers = {}

for conversation in conversations:
    cat = conversation['tag']
    category_answers[cat] = conversation['responses']

pickle.dump(category_answers, open('category_answers.pkl', 'wb'))

# print(category_answers)

# Procesamiento de documentos de preguntas

# Carga el modelo de lenguaje en español de SpaCy
nlp = spacy.load('es_core_news_sm')

import itertools

# Lista para extraer preguntas
questions = []

for script in conversations:

    # Extrae las preguntas
    question = script['patterns']
    questions.append(question)

# Consolida todas las preguntas
documents = list(itertools.chain.from_iterable(questions))

# print(pd.DataFrame(conversations).explode('patterns'))

# pprint(documents)

# Limpiar los signos de puntuacion y lematizar las palabras
questions_processed = []

for doc in documents:
    tokens = nlp(doc)
    # Remueve signos de puntuacion y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]

    # Pasa a minusculas
    new_tokens = [t.lower() for t in new_tokens]

    # Une los tokens procesados en un string con espacios
    questions_processed.append(' '.join(new_tokens))

# pprint(questions_processed)

# Paso auxiliar para extraer las categorias usando pandas y explode

# Lee al archivo de script de conversaciones
# y lo extiende por el contenido de patterns

df_conversations = pd.DataFrame(conversations).explode(
    ['patterns']
).reset_index(drop=True)

# print(df_conversations[['patterns', 'tag']])

### Representacion numerica de documentos de preguntas ###

# Importamos CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instanciamos el vectorizador
vectorizer = CountVectorizer()

# Crea la bolsa de palabras con la lista de documentos
X = vectorizer.fit_transform(questions_processed)

bow = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Guardarlo en formato pickle para usarlo despues

# Salva el transformador que crea la bolsa de palabras
# con el texto limpio de las preguntas

pickle.dump(
    vectorizer,
    open('sample_vectorizer_bow.pkl', 'wb')
)

# Toma una muestra de 10 renglones de la bolsa de palabras:
# print(bow.sample(10))

# Une la bolsa de palabras de cada pregunta con su categoria
processed_data = pd.concat(
    [bow,
     df_conversations[['tag']]
    ], axis = 1
)

processed_data.info()

# Ordenna aleatoriamente el dataframe:
processed_data = processed_data.sample(
    frac=1,
    random_state=123
).reset_index(drop=True)

# One hot encoding de las categorias

# Obtenemos la represenacion one hot encoding de las categorias
# con la funcion get_dummies de pandas
# print(pd.get_dummies(processed_data['tag'], dtype='int'))

# Guardamos la lista de los encabezados de la tabla
sample_categories = list(pd.get_dummies(processed_data['tag']).columns)

# salvaremos la variable `sample_categories` en formato pickle
pickle.dump(
    sample_categories,
    open('sample_categories.pkl', 'wb')
)

### Entrenamiento del Modelo ###

# Dimension de datos de entrada (nube de palabras)
dim_x = len(processed_data._get_numeric_data().to_numpy()[0])

# Dimension de la representacion One Hot Encoding
dim_y = len(pd.get_dummies(processed_data['tag'], dtype = int).to_numpy()[0])

# Creamos un modelo en TensorFlow
model = Sequential([
    Dense(25, input_shape=(dim_x,), activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dropout(0.2),
    Dense(dim_y, activation='softmax')
])

# Compilamos el modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# Entremanos el modelo

# Bolsa de palabras como arreglo de numpy
X_train = processed_data._get_numeric_data().to_numpy()

# Representacion en One Hot Encoding de las categorias
y_train = pd.get_dummies(processed_data['tag'], dtype=float).to_numpy()

hist = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=100,
    verbose=1
)

pickle.dump(
    model,
    open('sample_model.pkl', 'wb')
)

### Prediciendo la respuesta para un nuevo mensaje ###

# Creamos una varibale ejemplo
new_menssage = "este?, ¿como te llamas!"

# Creamos una funcion para procesar el texto del nuevo mensaje
def text_pre_process(message: str):
    """
    Procesa el texto del nuevo mensaje
    """
    # Procesa el mensaje con SpaCy
    tokens = nlp(message)

    # Remueve signos de puntuacion y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]

    # Pasa a minusculas
    new_tokens = [t.lower() for t in new_tokens]

    # Une los tokens procesados con un espacio
    clean_message = ' '.join(new_tokens)

    return clean_message

# Llama a la funcion y procesa el texto del nuevo mensaje
# print(text_pre_process(new_menssage))

# Creamos una funcion para obtener la representacion de la nube de palabras
def bow_representation(message: str) -> np.array:
    """
    Obtenemos la representacion del mensaje en su
    codificacion de la nube de palabras
    """

    bow_message = vectorizer.transform(
        [message]
    ).toarray()

    return bow_message

# Llama a la funcion y obten la representacion de la nube de palabras
# print(bow_representation(text_pre_process(new_menssage)))

# print(model(bow_representation(text_pre_process(new_menssage))).numpy())
# print(sample_categories)

# Creamos una funcion para obtener la prediccion de la categoria
def get_prediction(bow_message: np.array) -> int:
    """
    Obtiene la prediccion de la categoria
    que corresponde al mensaje
    """

    # Calcula el indice entero al que corresponde la categoria
    prediction = model(bow_message).numpy()

    # Obtiene el indice de la entrada con probabilidad mayor
    index = np.argmax(prediction)

    predicted_category = sample_categories[index]

    return predicted_category

# print(get_prediction(bow_representation(text_pre_process(new_menssage))))

# Creamos una funcion para obtener la respuesta
def get_answer(category: str) -> str:
    """
    Obtiene el mensaje de respuesta para una categoria
    """

    # Obtiene las respuestas de la categoria
    answers = category_answers[category]

    # Selecciona una respuesta al azar
    ans = random.choice(answers)

    return ans

# Nos da la respuesta del bot
bot_answer = get_answer(
    get_prediction(
        bow_representation(
            text_pre_process(new_menssage)
        )
    )
)

print("Usuario: ", new_menssage)
print("ChatBot: ", bot_answer)