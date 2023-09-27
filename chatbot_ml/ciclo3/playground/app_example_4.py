"""
Aplicacion que crea un API REST para obtener
las prediccion de una regresion lineal
previamente entrenada
"""
#
import numpy as np
import spacy
import random
import pickle

# Flask
from flask import Flask, request, render_template
import pickle


# Cargamos el modelo de lenguaje en español de Spacy
nlp = spacy.load('es_core_news_sm')

# ---- archivos del procesamiento del modelo ----- #

# diccionario de respuestas del bot por categoria
sample_category_answers = pickle.load(
    open('category_answers.pkl','rb')
    )

# Transformador para texto a nube de palabras 
sample_vectorizer_bow = pickle.load(
    open('sample_vectorizer_bow.pkl','rb')
    )

# Lista ordenada de las categorias de los tags a predecir
sample_categories = pickle.load(
    open('sample_categories.pkl','rb')
    )

# Modelo previamente entrenado
# model = pickle.load(
#    open('sample_model.pkl','rb')
#     )

# Usar mejor exportarlo e inportarlo con keras
from keras.models import load_model

model = load_model("model_conversations_chatbot.h5")

# --------- Funciones auxiliares --------------- #


def text_pre_process(message: str):
    """
    Procesa el texto del nuevo mensaje
    """
    # Procesa el mensaje con spaCy
    tokens = nlp(message)
    # remueve signos de puntuacion y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]

    # pasa a minusculas
    new_tokens = [t.lower() for t in new_tokens]

    # une los tokens procesados con un espacio
    clean_message = ' '.join(new_tokens)

    return clean_message


def bow_representation(message: str) -> np.array:
    """
    Obtiene la representacion del mensaje en su
    codificacion de la nube de palabras
    """
    bow_message = sample_vectorizer_bow.transform(
        [message]
        ).toarray()

    return bow_message


def get_prediction(
        bow_message: np.array
        ) -> int:
    """
    Obtiene la prediccion de la categoria
    que corresponde al mensaje
    """

    # Calcula el indice entero al que corresponde la catforia
    prediction = model(bow_message).numpy()
    #prediction = model(bow_message)

    # Obtiene el indice de la entrada con probabilidad mayor
    index = np.argmax(prediction)

    predicted_category = sample_categories[index]

    return predicted_category


def get_answer(category: str) -> str:
    """
    Obtiene el mensaje de respuesta para una categoria
    """
    # Obtiene las respuestas de la categoria
    answers = sample_category_answers[category]

    # Selecciona una respuesta al azar
    ans = random.choice(answers)

    return ans

# --------- Aplicacion en Flask ------------------- #
# Instancia la app
app = Flask(__name__, template_folder='.')

# Define una ruta para saber si la aplicacion 
# esta corriendo
@app.route('/healtCheck')
def index():
    return "true"

# Define la interaccion entre la persona y el ChatBot
# a partir de sitio web del archivo index.html

# lista para guardar conversaciones
conversations = []


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.form['question']:
        # Procesa la pregunta para obtener la prediccion
        # del modelo
        raw_question = request.form['question']
        clean_question = text_pre_process(raw_question)
        bow_question = bow_representation(clean_question)
        prediction = get_prediction(bow_question)
        bot_answer = get_answer(prediction)

        # Crea textos de respuesta y pregunta
        question = 'Usuario: ' + raw_question
        answer = 'ChatBot: ' + bot_answer

        # Guarda los textos de conversacion en la lista
        conversations.append(question)
        conversations.append(answer)

        # comunica los textos de conversacion al archivo html
        return render_template('index.html', chat=conversations)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
