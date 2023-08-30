import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importa clase de Python para instanciar el modelo KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

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

# Variable indicadora de si el dia es fin de semana
df['is_weekend'] = np.where(df['date'].dt.weekday> 4,1,0)

# --------- División de datos en entrenamiento y prueba ------------

# datos ordenados
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

# Datos de entrenamiento
X_train = X.loc[: X.shape[0]-1440,:].drop(target_col, axis=1)
y_train = X.loc[: X.shape[0]-1440,:][target_col]

# Datos de prueba
X_test = X.loc[X.shape[0]-1440+1:,:].drop(target_col, axis=1)
y_test = X.loc[X.shape[0]-1440+1:,:][target_col]

print(X_train)

# ------------------- Modelos Basicos de aprendizaje de Maquina ------------------- #

# Modelos de Regresion Lineal y Regularizacion

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Instanciamos los diferentes modelos
lr = LinearRegression()
lr_lasso = Lasso(alpha=0.5)
lr_ridge = Ridge(alpha=0.5)
lr_elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)

models_regresion = {
    'Linear Regression': lr,
    'Lasso Regression': lr_lasso,
    'Ridge Regression': lr_ridge,
    'Elastic Net Regression': lr_elastic
}

for model_name in models_regresion.keys():
    print('Modelo: ', model_name)
    model = models_regresion[model_name]

    # Ajusta el modelo con los datos de prueba
    model.fit(X_train[weather_cols], y_train)

    y_train_pred = model.predict(X_train[weather_cols])
    y_test_pred = model.predict(X_test[weather_cols])

    # Error en conjunto de entrenamiento y prueba
    error_train = mean_squared_error(y_train, y_train_pred)
    error_test = mean_squared_error(y_test, y_test_pred)

    # errores
    print("Error RSME en train:", round(error_train,2) )
    print("Error RSME en test:", round(error_test,2) )

    print("----------------------------------------------")

'''
**Preguntas**

* En los resultados se aprecia que algunos modelos con regulzación tienen 
  mejores resultados que la regresión ordinaria.
* ¿Qué estrategia se podría elegir para encontrar valores de la regulzación 
  que mejoren el desempeño del modelo en los datos de prueba?
  R: Se puede hacer una busqueda para encontrar el mejor valor de alpha
'''

# ------------------- Arboles de Decision & Bosques Aleatorios ------------------- #

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Instanciamos los diferentes modelos
dt_1 = DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=5
    )

dt_2 = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=7,
    min_samples_leaf=6
)

dt_3 = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=25
)

rf = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_depth=10, min_samples_split=5,
    min_samples_leaf=2
)

models_trees_regresion = {
    'Decision_Tree_1': dt_1,
    'Decision_Tree_2': dt_2,
    'Decision_Tree_3': dt_3,
    'Random_Forest': rf
}

for model_name in models_trees_regresion.keys():

    print('Modelo: ', model_name)
    model = models_trees_regresion[model_name]

    # Ajusta el modelo con los datos de prueba
    model.fit(X_train[weather_cols], y_train)

    y_train_pred = model.predict(X_train[weather_cols])
    y_test_pred = model.predict(X_test[weather_cols])

    # Error en conjunto de entrenamiento y prueba
    error_train = mean_squared_error(y_train, y_train_pred)
    error_test = mean_squared_error(y_test, y_test_pred)

    # errores
    print("Error RSME en train:", round(error_train,2))
    print("Error RSME en test:", round(error_test,2))

    print("----------------------------------------------")

'''
**Preguntas**
* ¿Cuál de los tres modelos tuvo mejor desempeño? ¿Porqué sudecio eso?
   R: El modelo de Random Forest tuvo mejor desempeño, ya que este modelo obtuvo un menor RSME en la prueba
   se puede considerar como que estubo mejor ajustado a los datos de prueba
'''

# ------------------- Validación Cruzada y Selección del Mejor Modelo ------------------- #

from sklearn.model_selection import TimeSeriesSplit

# Datos para probar los indices de TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [3, 4], [1, 2], [3, 4], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Instanciamos la validaación cruzada para TimeSeriesSplit
# Notas: se generan 4 conjuntos con datos de prueba de 
# tamaño 2 hacie el futuro

tscv = TimeSeriesSplit(n_splits=4, test_size=2)

# Imprime los valores de los indices
for train_index, test_index in tscv.split(X):
    print('Entrenamiento: ', train_index, 'Pruebas: ', test_index)

# ---------------Calibración de hiper parámetros con validación cruzada----------------- #

# Ejemplo de validacion cruzada

'''
En este ejemplo ajustaremos un modelo de regresión de Lasso, como sabemos toma un hiper parámetros 
de nombre alpha, con la métrica RMSE. El conjunto de validación será de dos meses, equivalente a 1440 
observaciones hacia el futuro.
'''
from sklearn.model_selection import GridSearchCV

# Modelo de regresión Lasso
reg_lasso = Lasso(
    alpha=0.0001,
    max_iter=3000,
    random_state=0
)

# Definimos un diccionario con los valores del hiper parámetro a probar

parameters = {
    'alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]
}

# Creamos el conjunto de indices para la validación cruzada
n_splits = 5
tscv = TimeSeriesSplit(n_splits, test_size=1440)

# Comunicamos esta información al GridSearchCV

model_lasso = GridSearchCV(
    reg_lasso,
    parameters,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv,
)

# Ahora entrenamos el modelo y lo evualamos

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

model_lasso.fit(X_train[weather_cols], y_train)

# Evaluamos el modelo

y_train_pred = model_lasso.predict(X_train[weather_cols])
y_test_pred = model_lasso.predict(X_test[weather_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(error_train,2))
print("Error RSME en test:", round(error_test,2))

print("Resultados")
print("Mejor modelo en la calibracion: \n", model_lasso.best_estimator_)
print("Mejor metrica de evaluacion: \n", model_lasso.best_score_)
print("Mejor valor de hiperparametro: \n", model_lasso.best_params_)

'''
**Preguntas**
* ¿Cuál fue el hiper parámetro que dió el mejor modelo?
    R: El mejor hiperparametro fue alpha = 1
'''

# ------------------- Validacion Cruzada con Pipelines ------------------- #

from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define listas de columnas que van a emplearse en el modelo
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

# Creamos el pipeline de pre-procesamiento

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standard_scaler', StandardScaler()),
    # ----------- Aquiseleccionamos las 4 mejores variables ------------ #
    ('select_k_best', SelectKBest(r_regression, k=4))
])

# Pipelien para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas especificadas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, seasons_cols)
], remainder='passthrough')

# Combina el pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('transform', pre_processor),
    # Define el modelo lasso
    ('model', Lasso(alpha=0.0001, max_iter=3000, random_state=0))
])

# Diccionario con el nombre del modelo
param_grid = {
    'model__alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]
}

lasso_reg_2 = GridSearchCV(
    pipe_standard_ohe,
    param_grid,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv,
)

lasso_reg_2.fit(X_train[non_target_cols], y_train)

# Evaluamos el modelo

y_train_pred = lasso_reg_2.predict(X_train[non_target_cols])
y_test_pred = lasso_reg_2.predict(X_test[non_target_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(error_train,2))
print("Error RSME en test:", round(error_test,2))

print("Resultados")
print("Mejor modelo en la calibracion: \n", lasso_reg_2.best_estimator_)
print("Mejor metrica de evaluacion: \n", lasso_reg_2.best_score_)
print("Mejor valor de hiperparametro: \n", lasso_reg_2.best_params_)

# ------------------- Ejemplo de Validacion Cruzada ------------------- #
#                   con Pipeline y Mas de un Modelo 

# Especificamos el pipeline de procesamiento

# Pipeline para escalar con estandar z-score
numerical_pipe = Pipeline([
    ('standard_scaler', StandardScaler()),
    # ----------- Aquiseleccionamos las 4 mejores variables ------------ #
    ('select_k_best', SelectKBest(r_regression, k=4))
])

# Pipelien para aplicar one hot encoding
categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina ambos procesos en columnas especificadas en listas
pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, seasons_cols)
], remainder='passthrough')

# Combina el pipeline la lista en el orden que se deben aplicar
# estos pasos

pipe_standard_ohe = Pipeline([
    ('pre_processor', pre_processor),
    # Define el modelo lasso
    ('model', RandomForestRegressor())
])

# Especificaciones de modelos y parametros

# Definimos los modelos a probar
model1 = Lasso()
model2 = RandomForestRegressor()

# Especificamos los hiperparametros del molelo 1
params1 = {}
params1['model__alpha'] = [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]
params1['model__max_iter'] = [1500, 3000]
params1['model'] = [model1] # <- Modelo dentro de una lista

# Modelo 2: Bosque Aleatorio

params2 = {}
params2['model__n_estimators'] = [3, 4, 5, 10],
params2['model__max_features'] = ["auto", "sqrt", "log2"]
params2['model__min_samples_split'] = [3, 4, 5, 10]
params2['model__bootstrap'] = [True, False]
params2['model'] = [model2] # <- modelo dentro de una lista

# Generamos una lista de los diccionarios de parametros y modelos

params_multi = [params1, params2]

model_csv_multi = GridSearchCV(
    pipe_standard_ohe,
    params_multi,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv=tscv,
)

# Realiza la transformacion de los datos y ajusta el modelo
model_csv_multi.fit(X_train[non_target_cols], y_train)

# Evaluamos el modelo

y_train_pred = model_csv_multi.predict(X_train[non_target_cols])
y_test_pred = model_csv_multi.predict(X_test[non_target_cols])

# Error en conjunto de entrenamiento y prueba
error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)

# errores
print("Error RSME en train:", round(error_train,2))
print("Error RSME en test:", round(error_test,2))

print("Resultados")
print("Mejor modelo en la calibracion: \n", model_csv_multi.best_estimator_)
print("Mejor metrica de evaluacion: \n", model_csv_multi.best_score_)
print("Mejor valor de hiperparametro: \n", model_csv_multi.best_params_)

'''
**Pregunta**
* ¿cual modelo tuvo mejor desempeño?
    R: El modelo que tuvo mayoy desempeño fue el Lasso
'''