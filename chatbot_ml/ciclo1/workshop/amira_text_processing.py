# ------------ librerias ------------ #

# Libreria de OS
import os
# Libreria de Numpy
import numpy as np
# Libreria de Spacy
import spacy
# Libreria de pandas
import pandas as pd
# Libreria de Seaborn
import seaborn as sns
# Libreria de JSON
import json
# Libreria de Matplotlib
import matplotlib.pyplot as plt
# Libreria de Wordcloud
from wordcloud import WordCloud
# Libreria de itertools
import itertools
# Libreria de Sklearn
from sklearn.feature_extraction.text import CountVectorizer
# Libreria de pickle
import pickle
# Libreria de warnings
import warnings
warnings.filterwarnings('ignore')

# ------------ Instrucciones ------------ #

'''
### 6. Entregables

A. Crea un script en Python (`amira_text_processing.py`) que procese los datos del 
	 archivo **conversations.json** para generar:
    A.1. Una lista llamada `vocabulario` con los resultados tokenizados y lematizados de las 
				 mensajes de entrada propuestas por el equipo de Amira, es decir, los textos de la clave`"patterns"`.
    A.2. Una lista llamada `tags` con los resultados de las etiquetas propuesta por el equipo 
				 de Amira, es decir, los textos de la clave`"tag"`.
    
En todos los casos estos resultados deberán:
1) ordenarse por orden alfabético, 
2) usar letras en minúsculas, 
3) eliminar signos de puntuación, así como los caracteres de interrogación y exclamación, 
4) no tener elementos duplicados.

B. El script anterior, deberá tener como outputs a dos archivos en formato pickle 
	 (`vocabulario.pkl` y `tags.pkl`, respectivamente). Esto se puede lograr con el código:

```
import pickle

# lista de vocabulario ya procesado
vocabulario = ...

# crea un objecto pickle del vocabulario limpio
pickle.dump(
    vocabulario,
    open('vocabulario.pkl', 'wb')
    )

# Este objeto se puede leer posteriormente como:
vocabulario = pickle.load(
    open('vocabulario.pkl', 'rb')
    )

```

C. En el mismo script, crea la bolsa de palabras de ejemplos de los mensajes que le podrían hacer 
llegar los usuarios (es decir el campo `patterns` dentro del archivo `conversations.json`). Como resultado 
de dicho proceso, se deberá crear una tabla en formato .csv que salve la bolsa de palabras de todos los 
mensajes con el nombre `bow_amira_patterns.csv`
'''

# Abre los datos de conversations.json
with open('conversations.json') as f:
    conversations = json.load(f)

# Cargamos el modelo de lenguaje en español de Spacy
nlp = spacy.load('es_core_news_sm')

# ------------ Vocabulario ------------ #
print("------------------ Vocabulario ------------------")
# Lista de vocabulario
vocabulario_gn = []

for script in conversations:
    messages = script['patterns']
    vocabulario_gn.append(messages)

# Unifica la lista de vocabulario
vocabulario_gn = list(itertools.chain.from_iterable(vocabulario_gn))

# Tokeniza el vocabulario
vocabulario_tok = []

for doc in vocabulario_gn:
    doc_tokens = nlp(doc)
    new_doc_tokes = [t.orth_.lower() for t in doc_tokens if not t.is_punct]
    vocabulario_tok.append(new_doc_tokes)

# Crea una lista con terminos limpios
vocabulario_tok_clean = list(
    itertools.chain.from_iterable(vocabulario_tok))

# Lemetiza el vocabulario
vocabulario_lem = []

for doc in vocabulario_tok_clean:
    doc_tokens = nlp(doc)
    new_doc_lem = [t.lemma_.lower() for t in doc_tokens]
    vocabulario_lem.append(new_doc_lem)

# Crea una lista con terminos limpios
vocabulario_lem_clean = list(
    itertools.chain.from_iterable(vocabulario_lem))

# Crea una lista de vocabulario sin duplicados
vocabulario = list(set(vocabulario_lem_clean))

# Ordena el vocabulario alfabeticamente
vocabulario.sort()
print(vocabulario)

# crea un objecto pickle del vocabulario limpio
pickle.dump(
    vocabulario,
    open('vocabulario.pkl', 'wb')
)

# ------------- tags ------------- #
print("------------------ Tags ------------------")
# Lista de tags
tags = []

for script in conversations:
    messages = script['tag']
    tags.append(messages)

print(tags)

pickle.dump(
    tags,
    open('tags.pkl', 'wb')
)

# ------------- BOW ------------- #
print("------------------ BOW ------------------")
# Instanciamos el transformador
vectorizer = CountVectorizer()

# Transferimos la variable que contiene el vocabulario_gn
vocabulario_bow = vocabulario_gn

# Crea la bolsa de palabras con la lista de vocabulario
X = vectorizer.fit_transform(vocabulario_bow)

# Crea un dataframe con la bolsa de palabras
bow = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
    )

print(bow)

# Guarda el dataframe en un archivo csv
bow.to_csv('bow_amira_patterns.csv', index=False)