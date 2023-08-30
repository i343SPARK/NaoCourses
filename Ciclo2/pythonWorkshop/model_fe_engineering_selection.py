# Librerias

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Pandas
import pandas as pd

# Numpy
import numpy as np

# SeaBorn
import seaborn as sns

# Matplotlib
import matplotlib.pyplot as plt

# Pickle
import pickle

# Sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder, LabelBinarizer, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, r_regression, VarianceThreshold

'''
Instrucciones:
A. Modificando el código anterior, crea un script que entrene varios modelos de regresión KNN, sobre 
los conjuntos de entrenamiento y pruebas descritos en este documento. El modelo deberá contar con las 
siguientes características:

    * Variables a considerar:
      * Variables numéricas del clima:
      * Variable categóricas (definiendo algun esquema de one-hot encoding):
        * seasons
        * holiday,
        * functioning_day
        * hour (para transformar a categoria puede usar el comando `data['hour].astype('category')`)
        * Las variables categóricas resultado del feature engineering
    * Feature enginering:
      * Crear una variables categóricas para representar: 1) el mes del año, 2) binaria que represente 
				si el día que transcurre es o no un fin de semana.
      * Transformar las variables numéricas del clima usando la transformación *Yeo-Johnson*
    * Feature selection: Realizar la selección de variable usando el criterio de VarianceThreshold
    * El modelo a seleccionar debe ser un KNN de 30 vecinos.
    * Se debe usar los transformadores de SKlearn y el pipeline para entrenar el modelo en el conjunto 
			de entrenamiento y probar su desempeño en el conjunto de prueba.
'''

# Detallar variables y archivos necesarios

# Variable que contendra la ruta del archivo de BikePro

route = 'SeoulBikeData.csv'

# Variable que usa pandas para leer el archivo

df = pd.read_csv(route, encoding='ISO-8859-1')

# Limpiamos las columnas del dataframe que sean de caracteres especiales

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

df.columns = clean_columns

# Damos formato a columnas de fecha formato datetime

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Agregamos las columnas month y is_weekend

df['month'] = df['date'].dt.month
df['month'] = df['month'].astype('category')

df['is_weekend'] = np.where(df['date'].dt.weekday> 4,1,0)
df['is_weekend'] = df['is_weekend'].astype('category')

# Ordenamos los datos

X = df.sort_values(['date', 'hour'])

# Columnas del clima

# weather_cols = [
#     'temperature', 
#     'humidity',
#     'humidity',
#     'wind_speed',
#     'visibility',
#     'dew_point_temperature',
#     'solar_radiation',
#     'rainfall',
#     'snowfall'
# ]

# Definimos la columna a predecir

target_col = ['rented_bike_count']

print(df.info())
# print(df[df['is_weekend']!=0].head(48))

# Dividimos los datos de entrenamiento y prueba

# Datos de entrenamiento
X_train = X.loc[: X.shape[0] - 1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0] - 1440,:][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0] - 1440:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0] - 1440:,:][target_col]

# print(X_test)

# //////Desarrollo de actividad del ciclo 2\\\\\\

# Define listas de columnas que van a emplearse en el modelado
seasons_cols = ['seasons']

weather_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall',
 ]

holiday_cols = ['holiday']

functioning_day_cols = ['functioning_day']

time_cols = ['hour']

month_cols = ['month']

weekend_cols = ['is_weekend']

# Lista que tiene todas los grupos de columnas
non_target_cols =  seasons_cols + weather_cols + holiday_cols + functioning_day_cols + time_cols + month_cols + weekend_cols

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler()),
    # ----------- Aqui seleccionamos las 4 mejores variables -------- #
    # ('select_k_best',v(r_regression, k=4) ),
    ('variance_threshold', VarianceThreshold(threshold=0.5)),
])

# Pipeline para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline para aplicar ordinal encoding

binari_pipe = Pipeline([
    ('binari', LabelBinarizer())
])

# Pipeline para aplicar PowerTransformer (Yeo-Johnson)
power_pipe = Pipeline([
    ('power_transformer', PowerTransformer(method='yeo-johnson'))
    ])


# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('categorical_seasons', categorical_pipe, seasons_cols),
    ('numerical_weather', power_pipe, weather_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_functioning_day', categorical_pipe, functioning_day_cols),
    ('categorical_time', categorical_pipe, time_cols),
    ('categorical_month', categorical_pipe, month_cols),
    ('catecorical_weekend', categorical_pipe, weekend_cols)
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    ('model', KNeighborsRegressor(n_neighbors=30))
])

# Realiza la transformacion de los datos y el ajuste del modelo
pipe_standard_ohe.fit(X_train[non_target_cols], y_train)

y_train_pred = pipe_standard_ohe.predict(X_train[non_target_cols])
y_test_pred = pipe_standard_ohe.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(error_train,2))
print("Error RSME en test:", round(error_test,2))

# pt = PowerTransformer(method='yeo-johnson')

# for col in weather_cols:
#     sns.distplot(
#         pt.fit_transform(X_train[[col]]),
#         ).set(title=f'Figura 3: Distribución de `{col}` transformada')

#     plt.show()

pickle.dump(pipe_standard_ohe, open('model_fe_engeneering_selection.pkl', 'wb'))