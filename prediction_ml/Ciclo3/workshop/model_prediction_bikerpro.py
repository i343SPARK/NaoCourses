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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge

# Warnings
import warnings
warnings.filterwarnings('ignore')

# ----------------- Instrucciones ----------------- #

'''
### 6. Entregables

A. Modificando el código anterior, crea un script que entrene los modelos KNN, regresión de Ridge y 
Bosque Aleatorio realizando validación cruzada de sus hiper-parámetros, sobre los conjuntos de entrenamiento 
y pruebas descritos en este documento. Se espera que realizando ingenieria de características, selección de 
variables y calibración de hiper-parámetros alcance un valor de RMSE cercano a las 200 unidades.

**Hint:** Puedes usar las variables y pre-procesamiento que quieras, pero estas son sugerencias de lo que tu 
pipeline podría integrar:

    * Variables:
      * Variables numéricas del clima (todas a excepción de `dew_point_temperature`)
      * Variable categóricas con one-hot encoding:
        * seasons
        * hour (para transforma a categoria puede usar el comando `data['hour].astype('category')`)
        * dia de la semana (revisa la sección de feature engineering y conviertela en categoria)
      * Variables categoricas binarias:
        * Holiday
        * functional day
        * indicador de fin de semana

Este script deberá deberá denominarse `model_prediction_bikerpro.py` y deberá salvar el modelo resultante en 
formato picke denominado `model_prediction_bikerpro.pk`

B. Este script deberá generar dos gráficas que comparen los valores reales de la demanda de rente de bicicletas 
de BikerPro, tanto en los conjuntos de entrenamiento como de prueba ambas en formato .png con lo nombres 
`comparative_actual_model_train_set.png` y `comparative_actual_model_test_set.png`
'''

# ------------------- Utilidades ------------------- #

# Damos una variable que contenga la ruta del archivo

file_bikepro = 'SeoulBikeData.csv'

# Leemos con pandas
df = pd.read_csv(file_bikepro, encoding = "ISO-8859-1")

# print(df.head(10))

# Limpiesa de nombre de columnas

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

# Actualizacion de varable de fecha a formato de fecha
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Definimos variables auxiliares relacionadas con el tiempo

# Define el dia de la semana como variable categorica
df['weekday'] = df['date'].dt.weekday
df['weekday'] = df['weekday'].astype('category')

# Define el mes  como variable categorica
df['month'] = df['date'].dt.month
df['month'] = df['month'].astype('category')

# Define el fin de semana como variable categorica
df['weekend'] = np.where(df['date'].dt.weekday> 4,1,0)
df['weekend'] = df['weekend'].astype('category')

# Variable indicadora de si el dia es fin de semana

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

#Variables categoricas binarias:

# columna objectivo a predecir
target_col = ['rented_bike_count']

# Instanciamos la validación cruazada para TimeSeriesSplit
# Notas: se generan 4 conjuntos con datos de prueba de 
# tamaño 2 hacie el futuro
tscv = TimeSeriesSplit(n_splits=4, test_size=2)

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]


# ------------------- Correcciones ------------------- #
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

binarizer_pipe = Pipeline([
    ('binarizer', LabelBinarizer())
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical_weather', numerical_pipe, weather_cols),
    ('categorical_season', categorical_pipe, seasons_cols),
    ('categorical_func_day', categorical_pipe, functional_day_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_weekday', categorical_pipe, weekday_cols),
    ('categorical_weekend', categorical_pipe, weekend_cols),
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define modelo KNN
    ('model', KNeighborsRegressor())
])

# Variables con los modelos a probar
model1 = KNeighborsRegressor()
model2 = Ridge()
model3 = RandomForestRegressor()

# Parametros del modelo KNN
params1 = {}
params1['model__n_neighbors'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
params1['model__weights'] = ['uniform', 'distance']
params1['model'] = [model1]

# Parametros del modelo Regresion de Ridge
params2 = {}
params2['model__alpha'] = [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]
params2['model'] = [model2]

# Parametros del modelo Bosque Aleatorio
params3 = {}
params3['model__n_estimators'] = [100, 200, 300, 400, 500]
params3['model__max_features'] = ["auto", "sqrt", "log2"]
params3['model__min_samples_split'] = [3, 4, 5, 10]
params3['model__bootstrap'] = [True, False]
params3['model__min_samples_leaf'] = [1, 2, 4]
params3['model'] = [model3]

# Lista de parametros
multi_params = [params1, params2, params3]

# Entrenamiento de los modelos
multi_params_model = GridSearchCV(
    pipe_standard_ohe,
    multi_params,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv
)

# Checar punto de checkpoint para ver si todo va bien
print('checkpoint')

# Realiza la transformacion de los datos y el ajuste del modelo
multi_params_model.fit(X_train[non_target_cols], y_train)

y_train_pred = multi_params_model.predict(X_train[non_target_cols])
y_test_pred = multi_params_model.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(np.sqrt(error_train),2))
print("Error RSME en test:", round(np.sqrt(error_test),2))

print("Mejores parametros: ", multi_params_model.best_params_)
print("Mejor Score: ", multi_params_model.best_score_)
print("Mejor Modelo: ", multi_params_model.best_estimator_)

# Graficas de comparacion de valores de entrenamiento y prueba

plt.figure(figsize=(15, 5))

plt.plot(y_train.reset_index(drop=True))
plt.plot(list(y_train_pred))
plt.title("comparative_actual_model_train_set")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_actual_model_train_set.png')

# Plotting graph of Actual (true) values and Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title(f"comparative_actual_model_test_set")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_actual_model_test_set.png')

pickle.dump(pipe_standard_ohe, open('model_prediction_bikerpro.pk', 'wb'))

'''
# ------------------- Entrenamiento de Modelo KNN ------------------- #
print('#---------------- Entrenamiento de Modelo KNN ----------------#')

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

binarizer_pipe = Pipeline([
    ('binarizer', LabelBinarizer())
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical_weather', numerical_pipe, weather_cols),
    ('categorical_season', categorical_pipe, seasons_cols),
    ('categorical_func_day', categorical_pipe, functional_day_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_weekday', categorical_pipe, weekday_cols),
    ('categorical_weekend', categorical_pipe, weekend_cols),
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define modelo KNN
    ('model', KNeighborsRegressor(n_neighbors=4))
])

# Diccionario con el nombre del modelo
param_grid = {
    'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'model__weights': ['uniform', 'distance'],
}

knn_model = GridSearchCV(
    pipe_standard_ohe,
    param_grid,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv
)

print('checkpoint')
# Realiza la transformacion de los datos y el ajuste del modelo
knn_model.fit(X_train[non_target_cols], y_train)

y_train_pred = knn_model.predict(X_train[non_target_cols])
y_test_pred = knn_model.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(np.sqrt(error_train),2))
print("Error RSME en test:", round(np.sqrt(error_test),2))

print("Mejores parametros: ", knn_model.best_params_)
print("Mejor Score: ", knn_model.best_score_)
print("Mejor Modelo: ", knn_model.best_estimator_)

# Graficas de comparacion de valores de entrenamiento y prueba

plt.figure(figsize=(15, 5))

plt.plot(y_train.reset_index(drop=True))
plt.plot(list(y_train_pred))
plt.title("comparative_KNN_train")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_KNN_train_set.png')

# Plotting graph of Actual (true) values and Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title(f"comparative_KNN_test")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_KNN_test_set.png')

# print(df.info())

# input('Presiona enter para continuar...')

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

binarizer_pipe = Pipeline([
    ('binarizer', LabelBinarizer())
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical_weather', numerical_pipe, weather_cols),
    ('categorical_season', categorical_pipe, seasons_cols),
    ('categorical_func_day', categorical_pipe, functional_day_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_weekday', categorical_pipe, weekday_cols),
    ('categorical_weekend', categorical_pipe, weekend_cols),
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define modelo Regresion de Ridge
    ('model', Ridge(alpha=1.0))
])

# Diccionario con el nombre del modelo
param_grid = {
    'model__alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]
}

# Usar RandomizedSearchCV en lugar de GridSearchCV
ridge_reg = RandomizedSearchCV(
    pipe_standard_ohe,
    param_distributions=param_grid,
    n_iter=20, # Número de iteraciones para la búsqueda aleatoria
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv,
    random_state=42
)

print('checkpoint')
# Realiza la transformacion de los datos y el ajuste del modelo
ridge_reg.fit(X_train[non_target_cols], y_train)

y_train_pred = ridge_reg.predict(X_train[non_target_cols])
y_test_pred = ridge_reg.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba

error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(np.sqrt(error_train),2))
print("Error RSME en test:", round(np.sqrt(error_test),2))

print("Mejores parametros: ", ridge_reg.best_params_)
print("Mejor Score: ", ridge_reg.best_score_)
print("Mejor Modelo: ", ridge_reg.best_estimator_)

# Graficas de comparacion de valores de entrenamiento y prueba

plt.figure(figsize=(15, 5))

plt.plot(y_train.reset_index(drop=True))
plt.plot(list(y_train_pred))
plt.title("comparative_ridge_regression_train")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_ridge_regression_train_set.png')

# Plotting graph of Actual (true) values and Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title(f"comparative_ridge_regression_test")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_ridge_regression_test_set.png')

# input('Presiona enter para continuar...')

# ------------------- Entrenamiento de Modelo Bosque Aleatorio ------------------- #
print('#---------------- Entrenamiento de Modelo Bosque Aleatorio ----------------#')

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

binarizer_pipe = Pipeline([
    ('binarizer', LabelBinarizer())
])

# Combina ambos procesos en columnas espeficadas en listas
pre_processor = ColumnTransformer([
    ('numerical_weather', numerical_pipe, weather_cols),
    ('categorical_season', categorical_pipe, seasons_cols),
    ('categorical_func_day', categorical_pipe, functional_day_cols),
    ('categorical_holiday', categorical_pipe, holiday_cols),
    ('categorical_weekday', categorical_pipe, weekday_cols),
    ('categorical_weekend', categorical_pipe, weekend_cols),
], remainder='passthrough')

# comunica al pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define modelo KNN
    ('model', RandomForestRegressor(n_estimators=100))
])

# Diccionario con el nombre del modelo
param_grid = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_features': ["auto", "sqrt", "log2"],
    'model__min_samples_split': [3, 4, 5, 10],
    'model__bootstrap': [True, False],
    'model__min_samples_leaf': [1, 2, 4]
}

rf_model = GridSearchCV(
    pipe_standard_ohe,
    param_grid,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv
)

print('checkpoint')
# Realiza la transformacion de los datos y el ajuste del modelo
rf_model.fit(X_train[non_target_cols], y_train)

y_train_pred = rf_model.predict(X_train[non_target_cols])
y_test_pred = rf_model.predict(X_test[non_target_cols])

# error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(np.sqrt(error_train),2))
print("Error RSME en test:", round(np.sqrt(error_test),2))

print("Mejores parametros: ", rf_model.best_params_)
print("Mejor Score: ", rf_model.best_score_)
print("Mejor Modelo: ", rf_model.best_estimator_)

# Graficas de comparacion de valores de entrenamiento y prueba

plt.figure(figsize=(15, 5))

plt.plot(y_train.reset_index(drop=True))
plt.plot(list(y_train_pred))
plt.title("comparative_random_forest_train")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f"comparative_random_forest_train_set.png")

# Plotting graph of Actual (true) values and Predicted values
plt.figure(figsize=(15, 5))

plt.plot(y_test.reset_index(drop=True))
plt.plot(list(y_test_pred))
plt.title(f"comparative_random_forest_test")
plt.legend(['Actual', 'Predicted'])
# Guardar la imagen en la carpeta practica_T1
plt.savefig(f'comparative_random_forest_test_set.png')

'''