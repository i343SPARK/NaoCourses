# Entregable con k=3,5,10,15,20,50,100, 300, 500, 1000

# Librerias de Trabajo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

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

# Datos ordenados
x = df.sort_values(['date', 'hour'])

# Datos de entrenamiento
X_train = x.loc[: x.shape[0]-1440,:][weather_cols]
y_train = x.loc[: x.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = x.loc[x.shape[0]-1440:,:][weather_cols]
y_test = x.loc[x.shape[0]-1440:,:][target_col]

# Instanciamos el modelo KNN

# Variable k = 3,5,10,15,20,50,100, 300, 500, 1000
k = [3,5,10,15,20,50,100, 300, 500, 1000]

for k in k:
    print("prueba con K = ", k)

    print("prueba con K = ", k)

    # Modelo para 5 vecinos mas cercanos
    model = KNeighborsRegressor(n_neighbors=k)

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
    # Error RSME en train con 3: 123500.35
    # Error RSME en train con 5: 153736.91
    # Error RSME en train con 10: 183054.41
    # Error RSME en train con 15: 195455.84
    # Error RSME en train con 20: 202640.4
    # Error RSME en train con 50: 228917.3
    # Error RSME en train con 100: 256709.94
    # Error RSME en train con 300: 302930.35
    # Error RSME en train con 500: 317392.05
    # Error RSME en train con 1000: 336150.02

    print("Error RSME en test: ", round(error_test, 2))
    # Error RSME en test con 3: 361621.39
    # Error RSME en test con 5: 343912.4
    # Error RSME en test con 10: 323035.16
    # Error RSME en test con 15: 317912.47
    # Error RSME en test con 20: 313366.41
    # Error RSME en test con 50: 317643.48
    # Error RSME en test con 100: 319125.99
    # Error RSME en test con 300: 320535.82
    # Error RSME en test con 500: 320107.57
    # Error RSME en test con 1000: 323270.32

    # Data de entrenamiento: valores reales vs predicciones

    # Plotting graph of Actual (true) values vs Predicted values
    plt.figure(figsize=(15, 5))

    plt.plot(y_train.reset_index(drop=True))
    plt.plot(list(y_train_pred))
    plt.title(f"error_knn_weather_K_{k}_train")
    plt.legend(['Actual', 'Predicted'])
    # Guardar la imagen en la carpeta practica_T1
    plt.savefig(f'error_knn_weather_K_{k}_train.png')

    # Plotting graph of Actual (true) values and Predicted values
    plt.figure(figsize=(15, 5))

    plt.plot(y_test.reset_index(drop=True))
    plt.plot(list(y_test_pred))
    plt.title(f"error_knn_weather_K_{k}_test")
    plt.legend(['Actual', 'Predicted'])
    # Guardar la imagen en la carpeta practica_T1
    plt.savefig(f'error_knn_weather_K_{k}_test.png')