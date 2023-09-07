from sklearn.model_selection import RandomizedSearchCV
# ------------------- Librerias ------------------- #

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
from sklearn.feature_selection import SelectKBest, r_regression, VarianceThreshold, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge

# Warnings
import warnings
warnings.filterwarnings('ignore')

# ------------------- Utilidades ------------------- #

# Damos una variable que contenga la ruta del archivo
file_bikepro = 'SeoulBikeData.csv'

# Leemos con pandas
df = pd.read_csv(file_bikepro, encoding="ISO-8859-1")

# Limpiesa de nombre de columnas
clean_columns = [
    x.lower().
    replace("(°c)", '').
    replace("(%)", '').
    replace(" (m/s)", '').
    replace(" (10m)", '').
    replace(" (mj/m2)", '').
    replace("(mm)", '').
    replace(" (cm)", '').
    replace(" ", '_')
    for x in df.columns
]

df.columns = clean_columns

# Actualizacion de variable de fecha a formato de fecha
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Definimos variables auxiliares relacionadas con el tiempo

# Define el dia de la semana como variable categórica
df['weekday'] = df['date'].dt.weekday
df['weekday'] = df['weekday'].astype('category')

# Define el mes como variable categórica
df['month'] = df['date'].dt.month
df['month'] = df['month'].astype('category')

# Define el fin de semana como variable categórica
df['weekend'] = np.where(df['date'].dt.weekday > 4, 1, 0)
df['weekend'] = df['weekend'].astype('category')

# Variable indicadora de si el día es fin de semana

# --------------- División de datos en entrenamiento y prueba ---------------- #

# datos ordenados
X = df.sort_values(['date', 'hour'])

# Variables categoricas con one-hot encoding:

# Columnas del clima
weather_cols = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'solar_radiation',
    'rainfall',
    'snowfall',
]
# Columnas de las estaciones
seasons_cols = ['seasons']

# Columnas de la hora
time_cols = ['hour']

# Columnas de los dias funcionales
functional_day_cols = ['functioning_day']

# Variables categoricas binarias:

# Columnas de los dias festivos
holiday_cols = ['holiday']

# Columnas de los dias de la semana
weekday_cols = ['weekday']

# Columnas de los dias de fin de semana
weekend_cols = ['weekend']

# Variables categoricas binarias:

# columna objetivo a predecir
target_col = ['rented_bike_count']

# Instanciamos la validación cruzada para TimeSeriesSplit
# Notas: se generan 4 conjuntos con datos de prueba de
# tamaño 2 hacia el futuro
tscv = TimeSeriesSplit(n_splits=4, test_size=2)

# Datos de entrenamiento
X_train = X.loc[: X.shape[0] - 1440, :].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0] - 1440, :][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0] - 1440 + 1:, :].drop(target_col, axis=1)
y_test = X.loc[X.shape[0] - 1440 + 1:, :][target_col]

# ------------------- Entrenamiento de Modelo Regresion de Ridge ------------------- #
print('#---------------- Entrenamiento de Modelo Regresion de Ridge ----------------#')

# Lista que tiene todas los grupos de columnas
non_target_cols = weather_cols + seasons_cols + time_cols + functional_day_cols + holiday_cols + weekday_cols + weekend_cols

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standar_scaler', StandardScaler()),
    # ----------- Aqui seleccionamos las 4 mejores variables -------- #
    ('select_k_best', SelectKBest(f_regression, k=4)),
])

# Pipeline para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline para aplicar LabelBinarizer a 'functioning_day'
binarizer_pipe = Pipeline([
    ('binarizer', LabelBinarizer())
])

# Combina ambos procesos en columnas especificadas en listas
pre_processor = ColumnTransformer([
    ('numerical_weather', numerical_pipe, weather_cols),
    ('categorical_season', categorical_pipe, seasons_cols),
    ('categorical_func_day', categorical_pipe, functional_day_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_weekday', categorical_pipe, weekday_cols),
    ('categorical_weekend', categorical_pipe, weekend_cols),
], remainder='passthrough')

# Comunica al pipeline la lista en el orden que se deben aplicar estos pasos
pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define modelo Regresion de Ridge
    ('model', Ridge(alpha=1.0))
])

# Diccionario con el nombre del modelo
param_grid = {
    'model__alpha': [1e-15, 1e-13, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 30, 40, 45, 50, 55, 60, 100, 0.0014]
}

# Usar RandomizedSearchCV en lugar de GridSearchCV
ridge_reg = RandomizedSearchCV(
    pipe_standard_ohe,
    param_distributions=param_grid,
    n_iter=20,  # Número de iteraciones para la búsqueda aleatoria
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv,
    random_state=42
)

print('checkpoint')
# Realiza la transformación de los datos y el ajuste del modelo con búsqueda aleatoria
ridge_reg.fit(X_train[non_target_cols], y_train)

y_train_pred = ridge_reg.predict(X_train[non_target_cols])
y_test_pred = ridge_reg.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(np.sqrt(error_train), 2))
print("Error RSME en test:", round(np.sqrt(error_test), 2))

print("Mejores parametros: ", ridge_reg.best_params_)
print("Mejor Score: ", ridge_reg.best_score_)
print("Mejor Modelo: ", ridge_reg.best_estimator_)
