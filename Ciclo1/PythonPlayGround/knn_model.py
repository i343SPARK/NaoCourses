# Fundamentos de Modelos de Aprendizaje de Máquina

# Librerias de Trabajo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Lectura de Datos
df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')
# print(df.head())

# Formato de columnas en crudo
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

# Asignamos los nuevos nombres de columnas para el analisis
df.columns = clean_columns

# Convertir al formato de fecha
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# df.info()

# Un modelo basico de aprendizaje de maquina para regresion

# Procedimientos para entrenar el modelo

'''
* Obtener los datos en forma tabular para las características y etiquetdas deseadas,
* Dividir el conjunto en entrenamiento y prueba,
* Ajustar el modelo KNN sobre el conjunto de entrenamiento, midiendo el error.
* Predecir sobre el conjunto de prueba y medir el error.
'''

# Obteniendo los datos para el modelo

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

# Columna objetiva a predecir
target_col = ['rented_bike_count']

x = df[weather_cols + target_col]

# print(x.head())

# Datos ordenados
x = df.sort_values(['date', 'hour'])

# Verificamos que los datos siguen el orden correcto
# print(x)
# print(x.shape[0]-168)

# Datos de entrenamiento
X_train = x.loc[: x.shape[0]-1440,:][weather_cols]
y_train = x.loc[: x.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = x.loc[x.shape[0]-1440:,:][weather_cols]
y_test = x.loc[x.shape[0]-1440:,:][target_col]

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Ajustar el modelo KNN sobre el conjunto de entrenamiento, midiendo el error.

# Flujo a seguir
'''
* Instanciar la clase del modelo de aprendizaje con los parámetros iniciales
* Ajustar el modelo sobre los datos con el método `.fit(data)`
* Predecir con el modelo entrenado en los datos de entrenamiento y prueba con `.predict(data)`
* Medir el error de ambas predicciones y evaluar.
'''

# Importamos la libreria de python para instanciar el modelo KNN

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Instanciamos el modelo KNN

# Modelo para 5 vecinos mas cercanos
model = KNeighborsRegressor(n_neighbors=5)

# AJustamos el modelo sobre los datos de entrenamiento
model.fit(X_train, y_train)

# Predecimos para los datos de entrenamiento y prueba
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Medimos el error

# Error en conjunto de entrenamiento
error_train = mean_squared_error(y_train, y_train_pred)

# Error en conjunto de prueba
error_test = mean_squared_error(y_test, y_test_pred)

print("Error RSME en train: ", round(error_train, 2))
# Error RSME en train: 153736.91

print("Error RSME en test: ", round(error_test, 2))
# Error RSME en test: 343912.4

# Data de entrenamiento: valores reales vs predicciones

# Plotting graph of Actual (true) values vs Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_train.reset_index(drop=True))
plt.plot(list(y_train_pred))
plt.title("Actual (true) vs Predicted values")
plt.legend(['Actual', 'Predicted'])
plt.show()

# Plotting graph of Actual (true) values and Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title("Actual (true) vs Predicted values")
plt.legend(['Actual', 'Predicted'])
plt.show()

'''
**Preguntas**

¿Son los datos del clima suficientes para predecir la demanda de bicletas con buen nivel de error?
No exactamente, ya que hay mas valores a considerar para predecir la demanda de bicicletas que no se 
estan considerando en el modelo.

¿Que se puede hacer para mejorar el desempeño del modelo?
Se puede mejorar el desempeño del modelo considerando mas variables que influyen en la demanda de bicicletas, 
como por ejemplo la hora del dia, el dia de la semana, el mes del año, etc.
'''

'''
### 6. Entregables

A. Modificando el código anterior, crea un script que entrene varios modelos de regresión KNN, considerando 
valores de k=3,5,10,15,20,50,100, 300, 500, 1000 con las mismas columnas del clima y la variable 
rented_bike_count. Este script deberá generar un gráfica en formato .png del **error RMSE obtenido en el conjunto** 
de prueba vs los valores de K, indicando cual es el valor donde se obtiene menor error, dicho archivo deberá denominarse 
`error_knn_weather.py` y la imagen `error_knn_weather.png`
'''