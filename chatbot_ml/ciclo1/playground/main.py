# -------------- Librerias -------------- #
import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# -------------- Elementos de Procesamiento de Lenguaje Natural -------------- #

## Normalización de Texto (Tokenización) con Spacy ##

# Cargamos el modelo de lenguaje en español de Spacy
nlp = spacy.load('es_core_news_sm')

# Creamos una variable de texto
news = "Los corredores fueron presurosos, hacia la meta, en aquella tarde soleada de 1985 en el estadio donde se llevaba acabo la maratón!"

# Calculamos la tokenización
new_tokens = nlp(news)

# Imprime el token y un indice que lo numera
'''
for i, token in enumerate(new_tokens):
    print(i, token)
'''

## Lematización ##
# Imprime el token y un indice que lo numera
'''
for i, token in enumerate(new_tokens):
    # Nota: se accede con el metodo .lemma_
    print(i, token.lemma_)
'''

# Comparacion de Tokenizacion y Lematizacion
'''
example = pd.DataFrame({
    'tokens': [x for x in new_tokens],
    'lemma': [x.lemma_ for x in new_tokens]
})

print(example)
'''

## Consideraciones sobre limpieza de texto y "stop words" ##

# Elimina puntuaciones:
# Tokens sin signos de puntuación
'''
tokens_palabras = [t.orth_ for t in new_tokens if not t.is_punct]

for token in tokens_palabras:
    print(token)
'''

# Elimina stop words:
# Tokens sin stop words
'''
tokens_lexicos = [t.orth_ for t in new_tokens if not t.is_stop]

for token in tokens_lexicos:
    print(token)
'''

# Elimina puntuaciones y stop words:
# Tokens sin signos de puntuación ni stopwords
'''
tokens_lexicos_not_stop = [t.orth_ for t in new_tokens if not t.is_punct | t.is_stop]

for token in tokens_lexicos_not_stop:
    print(token)
'''

# -------------- Representaciones graficas -------------- #
## Graficos de frecuencias ##

# Abriendo datos de conversaciones json
with open('conversations.json') as f:
    conversations = json.load(f)

# print(conversations)

# Procesando las respuestas a los usuarios
horoscopos = []

for script in conversations:
    message = script['responses']
    horoscopos.append(message)

# print(horoscopos)

# Unificamos los mensajes en una sola lista con itertools
import itertools
documents = list(itertools.chain.from_iterable(horoscopos))

# print(documents[:15])

# Pasar a minusculas y limpieza de signos de puntuación y stopwords

# Genera la version de los documentos en tokens
horoscopos_tokens = []

for doc in documents:
    doc_tokens = nlp(doc)
    new_doc_tokens = [t.orth_.lower() for t in doc_tokens if not t.is_punct | t.is_stop]
    horoscopos_tokens.append(new_doc_tokens)

# Crea una lista con terminos limpios
horoscopos_tokens_clean = list(
    itertools.chain.from_iterable(horoscopos_tokens)
)

# print(horoscopos_tokens_clean[:20])

# Contamos la frecuencia de las palabras

# Crea un diccionario para guardar conteos
word_count = {}

# Cuenta palabras añadiendolas a un diccionario
for palabra in horoscopos_tokens_clean:
    if palabra in word_count.keys():
        word_count[palabra] += 1
    else:
        word_count[palabra] = 1

# print(word_count)

# Creamos una tabla en pandas con la información
df = pd.DataFrame(
    {'palabra': word_count.keys(),
     'frecuencia': word_count.values()}
)

df.sort_values(['frecuencia'], ascending=False, inplace=True)

# print(df.head(20))

# Creamos un grafico de frecuencias de palabras
'''
sns.barplot(
    data=df.head(20),
    y='palabra',
    x='frecuencia',
).set(title='Figura 1: Conteos de frecuencias de palabras en respuestas de Amira')

plt.show()
'''

'''
**Preguntas:**

* ¿Cuales son las palabras más populares?
Dentro de un top 5 son: 
- Color
- Trabajo
- Momento
- Salud
- Suerte

* ¿Hay temas de los que se habla repetidamente?, De ser 
el caso, ¿cuáles son esos temas?
Si, los temas más repetidos son:
- Trabajo
- Salud
- Fortuna
- Amor
- Condicion fisica
'''

## Nube de palabras ##
from wordcloud import WordCloud

# Crea nube de palabras con el conteo de palabras
'''
wc_freq = WordCloud(
    width = 500,
    height = 500,
    background_color = 'white',
    colormap = "magma"
).generate_from_frequencies(word_count)

# Eliminar los ejes y mostrar los datos como imagen
plt.axis('off')
plt.imshow(wc_freq, interpolation='bilinear')
plt.title('Figura 2: Nube de palabras a partir de frecuencias')

plt.show()
'''

# Uso de cadena de texto para generar nube de palabras

# Une las palabras insertando un espacio entre ellas a partir
# de los terminos que limpiamos

# print(" ".join(horoscopos_tokens_clean))

# Crea la nube de palabras con la cadena de texto
'''
wc = WordCloud(
    width = 500,
    height = 500,
    background_color = 'white',
).generate(" ".join(horoscopos_tokens_clean))

# Eliminar los ejes y mostrar los datos como imagen
plt.axis('off')
plt.imshow(wc, interpolation='bilinear')
plt.title('Figura 3: Nube de palabras a partir de cadena de texto')

plt.show()
'''

'''
**Preguna**

* ¿qué temas son notorios en las nubes de palabras anteriores?
En una seleccion de 5 palabras son:
- Trabajo
- Salud
- Momento
- Color
- Amor
'''

# -------------- Encajes de texto y Bolsas de Palabras -------------- #

## Bolsa de Palabras con SKLearn ##
from sklearn.feature_extraction.text import CountVectorizer

# Instancia el transformador
vectorizer = CountVectorizer()

# Crea la bolsa de palabras con la lista de documentos
X = vectorizer.fit_transform(documents)

bow = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# print(bow)
# Representacion del mensaje "Vuelve pronto, te mando un abrazo"

print(bow.query("abrazo == 1"))