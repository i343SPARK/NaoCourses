# ------------------ Librarias ------------------ #

# Libreria de os
import os

# Libreria de numpy
import numpy as np

# Libreria de spacy
import spacy

# Libreria de pandas
import pandas as pd

# Libreria de seaborn
import seaborn as sns

# Libreria de Json
import json

# Libreria de random
import random

# Libreria de pickle
import pickle

# Libreria de itertools
import itertools

# Libreria de sklearn
from sklearn.feature_extraction.text import CountVectorizer

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Libreria de pprint
from pprint import pprint

# Libreria de warnings
import warnings
warnings.filterwarnings('ignore')

# -------------------- Instrucciones --------------------- #
"""
### 6. Entregables

A. Modifica el código presentado anteriormente para crear un script en 
Python (`conversations_train.py`) que entre un modelo de Deep Learning, con el 
propósito de obtener mensajes de respuesta a las conversaciones especificadas 
en el archivo **conversations.json** provisto por el equipo de Amira Rashid, 
**asegurándote de remover signos de puntuacios y stop words**.

Este modelo deberá tener como elemento de salida, archivos de los siguiente elementos:

    * Un diccionario donde las llaves son categorías de preguntas y los 
			valores son las lista de respuestas que el equipo de la influencer 
			espera que el ChatBot comunique a sus usuarios, denomiando 
			`conversations_category_answers.json`,

    * El tranformador de Sklearn que permite transformar un mensaje en texto a la bolsa 
			de palabras, mismo que deberá ser ajusta con el corpus de preguntas de los usuarios, que 
			el equipo de Amira proporcinó. Este se deberá denominar `conversations_vectorizer_bow.pkl`

    * La versión pickle de la lista de nombres de categorías que asociadas a las preguntas 
			planteadas por el equipo de la influencer, en el orden específico en que se haya realizado 
			el One Hot Encoding para representarlas en el procesamiento del modelo. Este se deberá 
			denominar `conversations_categories.pkl`.

    * El modelo entrenado para tal efecto, en versión pickle. Dicho archivo deberá denominarse 
			`model_conversations_chatbot.pkl`

B. Adicionalmente, tras su ejecución el script deberá escribir a pantalla (estandar output, en consola), 
el mensaje que obtendría un usuario al escribir "Mi signo es Tauro", como resultado de emplear el modelo 
entrenado para predecir que mensaje asociarle.
"""

# Carga los datos
with open('conversations.json') as f:
    conversations = json.load(f)
    
# pprint(conversations)

# ------------------ Primer punto ------------------ #

# Crea un diccionario de tags como llaves con valores con las respuestas
category_answers = {}

for conversation in conversations:
    cat = conversation['tag']
    category_answers[cat] = conversation['responses']
    
# pprint(category_answers)

# Crea un archivo json con las respuestas
with open('conversations_category_answers.json', 'w') as f:
	json.dump(category_answers, f, indent=4)

# pickle.dump(category_answers, open('category_answers.pkl', 'wb'))
     
# ------------------ Segundo punto ------------------ #

# Carga el modelo de lenguaje en español de SpaCy
nlp = spacy.load('es_core_news_sm')

# Lista para extraer preguntas
questions = []

for script in conversations:

    # Extrae las preguntas
    question = script['patterns']
    questions.append(question)
    
# Juntamos las preguntas en una sola lista
documents = list(itertools.chain.from_iterable(questions))

# pprint(documents)

# Limpiamos los signos de puntuación y las stop words
questions_processed = []

for docs in documents:
    tokens = nlp(docs)
    # Remueve signos de puntuacion y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct | t.is_stop]

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

# Instanciamos el vectorizador
vectorizer = CountVectorizer()

# Crea la bolsa de palabras con la lista de documentos
X = vectorizer.fit_transform(questions_processed)

bow = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)


# Salva el transformador que crea la bolsa de palabras
# con el texto limpio de las preguntas

pickle.dump(
    vectorizer,
    open('conversations_vectorizer_bow.pkl', 'wb')
)

# print(bow)

# ------------------ Tercer punto ------------------ #

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
    open('conversations_categories.pkl', 'wb')
)

# ------------------ Cuarto punto ------------------ #

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
    optimizer='adam',
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
    open('model_conversations_chatbot.pkl', 'wb')
)

model.save("model_conversations_chatbot.h5")

# ------------------ Punto B ------------------ #

# Creamos una varibale ejemplo
new_message = "Mi signo es Tauro"

# Creamos una funcion para procesar el texto del nuevo mensaje
def text_pre_process(message: str):
    """
    Procesa el texto del nuevo mensaje
    """
    # Procesa el mensaje con SpaCy
    tokens = nlp(message)

    # Remueve signos de puntuacion y los stop words, y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct | t.is_stop]

    # Pasa a minusculas
    new_tokens = [t.lower() for t in new_tokens]

    # Une los tokens procesados con un espacio
    clean_message = ' '.join(new_tokens)

    return clean_message

# Llama a la funcion y procesa el texto del nuevo mensaje
print(text_pre_process(new_message))

# Creamos una funcion para obtener la representacion de la nube de palabras
def bow_representation(message: str) -> np.array:
    """
    Obtiene la representacion del mensaje en su
    codificacion de la nube de palabras
    """

    bow_message = vectorizer.transform(
        [message]
        ).toarray()

    return bow_message

# Llama a la funcion y obten la representacion de la nube de palabras
print(bow_representation(text_pre_process(new_message)))

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

    # print(prediction)

    # Obtiene el indice de la entrada con probabilidad mayor
    index = np.argmax(prediction)

    # print(index)

    predicted_category = sample_categories[index]

    # print(predicted_category)

    return predicted_category

print(get_prediction(bow_representation(text_pre_process(new_message))))

# print(sample_categories)

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
            text_pre_process(new_message)
        )
    )
)

print("Usuario: ", new_message)
print("ChatBot: ", bot_answer)