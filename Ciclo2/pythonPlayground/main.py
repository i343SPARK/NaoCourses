import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importa clase de Python para instanciar el modelo KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

FILE_BIKEPRO = 'SeoulBikeData.csv'

# Creamos el dataframe con los datos
df = pd.read_csv(FILE_BIKEPRO, encoding='ISO-8859-1')

# print(df.head(10))

# Limpiamos las columnas para mayor facilidad de uso
raw_columns = list(df.columns)

clean_columns = [
    x.lower().\
        replace("(°c)", '').\
        replace("(%)", '').\
        replace(" (m/s)", '').\
        replace(" (10m)", '').\
        replace(" (mj/m2)", '').\
        replace("(mm)", '').\
        replace(" (cm)", '').\
        replace(" ", '_')
    for x in df.columns
    ]

# Asignamos los nuevmos nombres de columnas para el análisis
df.columns = clean_columns

# Convertimos las columnas de fecha a formato de fecha
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# print(df.info())

# Ordenamos los datos:

X = df.sort_values(['date', 'hour'])

# Columnas del clima
weather_cols = [
    'temperature', 
    'humidity',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall'
    ]

# columna objectivo a predecir
target_col = ['rented_bike_count']

# Dividimos los datos en entrenamiento y prueba

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de entrenamiento
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

# print(X_test)

# Variables Polinomicas e interacciones

'''
Pasos a seguir:
* Ajustar el transformador `.fit` en el **conjunto de entrenamiento**
* Obtener los datos transformados `.predict` para el conjunto de entrenamiento y prueba,
* Continuar con el flujo de ajuste del modelo.

**Nota:** Los transformadores **siempre** deben ajustarse en el conjunto de entrenamiento, 
        dado que usar el conjunto de prueba o en los datos originales completos puede producir 
        fuga de datos.
'''

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly.fit(X_train[weather_cols])

X_train_poly = pd.DataFrame(
    poly.transform(X_train[weather_cols]),
    columns=poly.get_feature_names_out(weather_cols)
)

X_test_poly = pd.DataFrame(
    poly.transform(X_test[weather_cols]),
    columns=poly.get_feature_names_out(weather_cols)
)

# print(X_train_poly.transpose())

# Entrenamos el modelo

# Instancia el modelo KNN para 5 vecinos
model_poly = KNeighborsRegressor(n_neighbors=5)

# Ajusta el modelo con los datos de prueba
model_poly.fit(X_train_poly, y_train)

y_train_pred = model_poly.predict(X_train_poly)
y_test_pred = model_poly.predict(X_test_poly)

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Sin Pipeline")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Error RSME en train:  160920.54
# Error RSME en test:  352379.85

'''
**Pregunta**

* El modelo presenta un error sustancialmente más alto en el conjunto 
de entrenamiento que en el de prueba ¿está subajustando o sobreajustando?
Esta Sobre-ajustado, ya que el error en el conjunto de entrenamiento es mayor 
que en el de prueba.

* Considerando lo anterior, ¿qué consecuencias tendria usar con un modelo 
con estas características para las predicciones de la demanda de BikerPro?
La consecuencia de usar un modelo sobreajustado para las predicciones de demanda 
de bicicletas sería que las predicciones serían poco confiables y probablemente 
no se generalizarían bien a nuevos datos.
'''

# Pipeline de Sklearn

from sklearn.pipeline import Pipeline

# Crea lista de tuplas con el nombre de transformaciones/modelo
# junto con sus nombres

estimators = [
    # Creacion de variables polinomicas
    ('polinomical_features', PolynomialFeatures(degree=2)),
    # Modelo KNN de 5 vecinos mas proximos
    ('knn_model', KNeighborsRegressor(n_neighbors=5))
]

# Comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe = Pipeline(steps=estimators)

# Realiza la transformacion de los datos y el ajuste del modelo
pipe.fit(X_train[weather_cols], y_train)

y_train_pred = pipe.predict(X_train[weather_cols])
y_test_pred = pipe.predict(X_test[weather_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Con Pipeline")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Error RSME en train:  160920.54
# Error RSME en test:  352379.85

# Normalizacion de la escala de variables

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Ejemplo de ajuste con escalamiento estandar
standard_scaler = StandardScaler()

# print(pd.DataFrame(
#     standard_scaler.fit_transform(X_train[weather_cols]),
#     columns=weather_cols
# ))

# Combinando transformaciones con Pipeline

# Crear lsita de tuplas con el nombre de transformaciones/modelo
# junto con sus nombres

estimators_standard_poly = [
    # Escalamiento estandar
    ('min_max_scaler', StandardScaler()),
    # Creacion de variables polinomicas
    ('polinomical_features', PolynomialFeatures(degree=2)),
    # Modelo KNN de 5 vecinos mas proximos
    ('knn_model', KNeighborsRegressor(n_neighbors=5))
]

# Comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos
pipe_standard_poly = Pipeline(steps=estimators_standard_poly)

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_standard_poly.fit(X_train[weather_cols], y_train)

y_train_pred = pipe_standard_poly.predict(X_train[weather_cols])
y_test_pred = pipe_standard_poly.predict(X_test[weather_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Con Pipeline y Escalamiento Estandar")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Error RSME en train:  116619.56
# Error RSME en test:  331116.32

# Crea lista de tuplas con nombre de transformaciones/modelo
# junto con sus nombres

estimators_min_max_poly = [
    # Escalamiento min max
    ('min_max_scaler', MinMaxScaler()),
    # Creacion de variables polinomicas
    ('polinomical_features', PolynomialFeatures(degree=2)),
    # Modelo KNN de 5 vecinos mas proximos
    ('knn_model', KNeighborsRegressor(n_neighbors=5))
]

# Comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_min_max_poly = Pipeline(steps=estimators_min_max_poly)

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_min_max_poly.fit(X_train[weather_cols], y_train)

y_train_pred = pipe_min_max_poly.predict(X_train[weather_cols])
y_test_pred = pipe_min_max_poly.predict(X_test[weather_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Con Pipeline y Escalamiento Min Max")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Error RSME en train:  112683.9
# Error RSME en test:  342655.27

# Transformaciones a variables categoricas

from sklearn.preprocessing import (LabelEncoder, 
                                   OrdinalEncoder, 
                                   LabelBinarizer, 
                                   OneHotEncoder)

# Label Encoding
# Usando el campo de Seasons

le = LabelEncoder()
# print(le.fit_transform(X_train['seasons']))
# print(le.classes_)

# Binary Encoding
# Sobre functioning_day

# Instanciamos la clase
lb = LabelBinarizer()

# print(lb.fit_transform(X_train['functioning_day']))
# print(lb.classes_)

# One Hot Encoding
# Sobre Seasons

ohe = OneHotEncoder()

# Ajusta el transformador para generar el encoding
transformed = ohe.fit(X_train[['seasons']])

# Crea un dataframe para visualizar el resultado del encoding

# print(pd.DataFrame(
#     transformed.transform(X_train[['seasons']]).toarray(),
#     columns=transformed.categories_
# ))

# Combinando procesamientos de variables numericas y categoricas:
# Pipeline y ColumnTransformer

from sklearn.compose import ColumnTransformer

# Define listas de columnas que van a emplearse en el modelado

weather_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall'
]

seasons_cols = ['seasons']

time_cols = ['hour']

# Lista que tiene todos los grupos de columnas
non_target_cols = weather_cols + seasons_cols + time_cols

# Creamos 2 pipelines para procesar a las variables numericas y categoricas

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standard_scaler', StandardScaler())
])

# Pipeline para codificar con One Hot Encoding
categorical_pipe = Pipeline([
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas especificas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, seasons_cols)
], remainder='passthrough')

# Este pre-procesamiento se puede incluir ahora en un pipeline con el ajuste de modelo:

# Comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    ('model', KNeighborsRegressor(n_neighbors=5))
])

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_standard_ohe.fit(X_train[non_target_cols], y_train)

y_train_pred = pipe_standard_ohe.predict(X_train[non_target_cols])
y_test_pred = pipe_standard_ohe.predict(X_test[non_target_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Con Pipeline y Escalamiento Estandar y One Hot Encoding")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Error RSME en train:  67848.04
# Error RSME en test:  244534.63

'''
**Preguntas**

* ¿Existe una mejora en el entrenamiento del modelo?
Si, el modelo predice en cierta forma mejor, ya que el 
error no es tan alto, como en los ejemplo anteriores.

* ¿Existe sub-ajuste o sobre ajuste del modelo?
En cierto modo si, seria de Sobreajuste, sin embargo, como el error bajo
podriamos que mejoro en este aspecto y a reducido este sobreajuste.
'''

# Ejemplos con algunas variables

# print(df['date'].dt.weekday)

df['weekday'] = df['date'].dt.weekday

# Fin de semana

# Crea variable indicadora de fin de semana
# 1 = fin de semana
# 0 = no fin de semana

df['is_weekend'] = np.where(df['date'].dt.weekday>4,1,0)

# print(df['is_weekend'])

'''
**Preguntas**

* ¿Cómo se crearía un variable representativa de mes?
Se crearia con una variable llamada 'month' y se pudiese 
representar con un numero del 1 al 12.

* ¿Es mejor representar dicha variable como numérica o categórica?
Es mejor representarla como numerica, ya que como son meses, tienen propiedades individuales
como los dias festivos o como los dias del mes
'''

# Transformacion de caracteristicas
# Cambios en la distribucion de los datos

# Ejemplo de cambio en la distribucion para solar_radiation

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()

# sns.displot(X_train['solar_radiation']).set(title='Distribucion de solar_radiation')

# plt.show()

# sns.displot(pt.fit_transform(X_train[['solar_radiation']])).set(title='Distribucion de solar_radiation')

# plt.show()

# Seleccion de caracteristicas
# Ejemplo de seleccion de variables a traves de SelectBest y r_regression

'''
Para ello seguiremos el ejemplo de tomar variables del clima, incluir 
la hora y procesar con one-hot encoding seasons para ajustar un modelo 
KNN, pero realizando selección de 4-mejores variables por el criterio 
de correlacion.
'''

from sklearn.feature_selection import SelectKBest, r_regression

# Define listas de columnas que van a emplearse en el modelado
weather_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall'
]

seasons_cols = ['seasons']

time_cols = ['hour']

# Lista que tiene todos los grupos de columnas
non_target_cols = weather_cols + seasons_cols + time_cols

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler()),
    # Aqui seleccionamos las 4 mejores variables
    ('select_k_best',SelectKBest(r_regression, k=4) ),
])

# Pipeline para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas especificas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, seasons_cols)
], remainder='passthrough')

# Comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    ('model', KNeighborsRegressor(n_neighbors=5))
])

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_standard_ohe.fit(X_train[non_target_cols], y_train)

y_train_pred = pipe_standard_ohe.predict(X_train[non_target_cols])
y_test_pred = pipe_standard_ohe.predict(X_test[non_target_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# Errores
print("Con Pipeline y Escalamiento Estandar y One Hot Encoding")
print('Error RSME en train: ', round(error_train, 2))
print('Error RSME en test: ', round(error_test, 2))

# Se puede salvar el modelo anterior usando la utilidad pickle de Python

import pickle
pickle.dump(pipe_standard_ohe, open('pipe_standard_ohe.pkl', 'wb'))

# Y se puede volver a leer con pickle.load

pickle_model = pickle.load(open('pipe_standard_ohe.pkl', 'rb'))