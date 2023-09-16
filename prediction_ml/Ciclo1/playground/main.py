import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')

raw_columns = list(df.columns)
# print(raw_columns)

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

# print(clean_columns)

# print(pd.DataFrame({'raw_columns': raw_columns, 'clean_columns': clean_columns}))

df.columns = clean_columns

# print(df.info())

'''
**Preguntas**

¿Cuántas columnas tiene la tabla?
12

¿Qué variables se relacionan con el tiempo?
date, hour, temperature, dew_point_temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, seasons

¿Es correcto que el formato de `date` sea del tipo `object`? Sino, ¿qué se debe hacer para corregirlo?
No, se debe convertir a datetime

¿Qué columnas se relacionan con variables del clima?
temperature, dew_point_temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall

¿Cuáles son indicadores de algún evento especial es un época del año?
seasons

¿Cuáles variables son numéricas y cuáles categóricas y porqué?
numéricas: temperature, dew_point_temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall
categóricas: date, hour, seasons
'''

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
# print(df.info())
# print(df.describe().transpose())
# print(df["date"].describe())

'''
**Preguntas**

Observando la tabla anterior:
¿Cuál es la cantidad promedio de bicicletas que se rentan por hora?
704.602055

¿Cuál es la hora promedio en que se rentan éstos dispositivos?
11.5

¿Qué significa que la variable `temperature` tenga un valor mínimo negativo?
Que la temperatura es negativa

¿qué puede esperarse de las condiciones climátics a esa temperatura y de la renta de bicicletas?
Que no se renten bicicletas

¿Cuál es el rango de fechas mínimo y máximo que abarcan los datos de BikePro?
2017-12-01 - 2018-11-30

¿La escala de las variables es la misma o existen algunas en ordenes de magnitud diferentes (e.g. miles vs decimales)?
No, no es la misma escala de las variables ya que algunas son de ordenes de magnitud diferentes
'''

# Variables relacionadas con la renta de bicicletas
#Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.histplot(data=df, x='rented_bike_count').set(title='Histograma de la cantidad de bicicletas rentadas')
# plt.show()

# sns.boxplot(data=df, x='rented_bike_count').set(title='Histograma de la cantidad de bicicletas rentadas')
# plt.show()

'''
**Preguntas**

¿Qué puede motivar los valores atípicos en la rentas de bicicletas?
Que haya un evento especial como un dia festivo o un dia de lluvia

¿Será necesario incoporar estos factores al modelo predictivo?
Si, ya que son factores que afectan la renta de bicicletas

¿Qué sucedería en el modelo predictivo si no se toman en cuenta tales factores?
Que el modelo no sea tan preciso y por lo tanto habria un margen de error notable
'''

# calcula los valore de renta de servicios agrupados por dia
temp_data = df[['date', 'rented_bike_count']].groupby('date').sum().reset_index()
# print(temp_data)

# convierte el dato a categorico para el plot
temp_data['date'] = temp_data['date'].astype('category')

# especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.lineplot(x='date', y='rented_bike_count', data=temp_data).set(title='Cantidad de bicicletas rentadas por dia')
# print(temp_data.describe().transpose())
# plt.show()

'''
**Preguntas**

¿A partir de que punto comenzó el incremento en la renta de servicios?
A partir del 2018-03-01

En el diagrama se observa que existe valores donde la renta de servicios cae de manera abrupta hasta ser cero. ¿Qué razones pueden explicar dichas caídas? 
Que haya un evento o que el clima no obtuvo las condiciones necesarias para rentar bicicletas

¿Se puede pensar que de un día para otro las personas dejaron de rentar bicicletas o existe algun otro factor?
Es probable que si, ya que puede haber un factor lluvia o evento que afecte la renta de bicicletas
'''

# Variables relacionadas con el clima

# Espeficicamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.histplot(data=df, x='temperature').set(title='Histograma de la temperatura')
# # plt.show()

# plt.figure(figsize=(15, 6))
# sns.boxplot(data=df, x='temperature').set(title='Boxplot de la temperatura')
# plt.show()

#Variables relacionados con la radiacion solar

# Especificar el tamaño de la figura
# plt.figure(figsize = (15, 6))

# sns.histplot(data=df, x= 'solar_radiation').set(title="Histograma de la radiacion solar")

# plt.show()

# plt.figure(figsize = (15, 6))

# sns.boxplot(data=df, x= 'solar_radiation').set(title="Boxplot de la radiacion solar")

# plt.show()

#Variables relacionadas con la visibilidad
# print(df['visibility'].describe())

# plt.figure(figsize = (15, 6))
# sns.histplot(data=df, x= 'visibility', log_scale=True).set(title="Histograma de la visibilidad")
# plt.show()

'''
**Pregunta**

* Ejecute el código 

```
    plt.figure(figsize = (20,10))
    sns.boxplot(data = bikerpro)
```
¿Representará un desafío para predecir la 
renta de bicicletas que el modelo tenga variables 
en diferentes escalas?

Si, ya que son varias variables que estan en diferentes escalas
'''

# plt.figure(figsize = (20, 10))
# sns.boxenplot(data = df)
# plt.show()

# Variables relacionadas con eventos de temporada

# Conteos de los registros por las categorias de seasons
# print(df['seasons'].value_counts())
# print(df['seasons'].value_counts(normalize=True))

# sns.countplot(data=df, x='seasons').set(title='Conteo de registros por temporada')
# plt.show()

'''
**Preguntas**

Si hacemos conteos de las categorías de la variable `holiday`, 
¿alguna categoría es mayor a otra o no?, ¿que significa la distribución que se obtiene?
Si, la categoria que es mayor es la de no festivo, lo cual significa que la mayoria de 
los dias no son festivos
'''

# Analisis Exploratorio Bivariado

# Rented_bike_count vs hour

# Suma la cantidad de bibicletas rentadas por hora
# temp_bike_hour = df.groupby('hour')['rented_bike_count'].sum().reset_index()

# # Convierte la hora a categoria para realizar el plot
# temp_bike_hour['hour'] = temp_bike_hour['hour'].astype('category')

# sns.barplot(x='hour', y='rented_bike_count', data=temp_bike_hour).set(title='Cantidad de bicicletas rentadas por hora')
# plt.show()

'''
**Preguntas**

Cuales son los horarios con mayor afluencia de renta de servicios de BikePro?
Las 8 y las 18 horas

En la imagen se visualizan dos picos, uno a las 8 y otro a las 16 horas ¿qué 
podría explicar estos eventos en el incremento de renta de servicios?
Que la gente renta bicicletas para ir a trabajar y para regresar a casa

¿Existe algun horario donde la renta de biclicetas tenga una baja sustancial? 
En caso afirmativo, ¿qué causa puede explicar ese fenómeno?
Si, a las 2 y a las 3 de la mañana, ya que la mayoria de la gente esta durmiendo
'''


# mean temperature vs date

# Calcula los valores de renta de servicios agrupados por dia
# temp_data_temperature = df[['date', 'temperature']].groupby('date').mean().reset_index()

# # Convierte el dato a categorico para el plot
# temp_data_temperature['date'] = temp_data_temperature['date'].astype('category')

# # Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.lineplot(x='date', y='temperature', data=temp_data_temperature).set(title='Temperatura promedio por dia')
# plt.show()

'''
**Preguntas**
¿A partir de que punto se observa un incremento en las temperaturas promedio?
A partir de mediados de Febrero del 2018

¿En que fecha se alcanzó el pico de temperatura observada?
Mediados de Julio del 2018

¿Cómo se comparan las observaciones anteriores sobre esta serie de tiempo 
con la Figura 3 que muestra la evolución de la demanda de servicios?
Se puede observar que la demanda de servicios aumenta conforme aumenta la temperatura

¿Existe alguna similitud entre el aumento de los servicios y el aumento de la temperatura?
Si, ya que la demanda de servicios aumenta conforme aumenta la temperatura
'''

# temperature vs rented_bike_count

# Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.scatterplot(x='temperature', y='rented_bike_count', data=df).set(title='Temperatura vs Cantidad de bicicletas rentadas')
# plt.show()

'''
**Preguntas**

¿Qué sucede con la demanda de bicletas cuando hay temperatuas bajas, aumenta o disminuye?
Disminuye

¿Se debe tomar en cuenta esta variable para desarollar un modelo predictivo de la demanda de BikePro?
Si, ya que la demanda de bicicletas depende tambien de la temperatura

¿qué sucedería con las predicciones de un modelo si no tomaramos en cuenta a la temperatura?
Que no serian tan precisas
'''

# Season vs rented_bike_count

# Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.boxplot(x='seasons', y='rented_bike_count', data=df).set(title='Temporada vs Cantidad de bicicletas rentadas')
# plt.show()

'''
**Preguntas**
¿Cúal es la estacion con menor demanda de servicios de BikerPro?
Invierno 

¿qué razones pueden estar detrás de la baja en la renta de bicletas?
Que hace mucho frio y la gente no quiere salir de sus casas

¿Cúal es la estacion con mayor demanda de servicios de BikerPro?
Verano

¿qué razones pueden estar detrás de la alta en la renta de bicletas?
Que los factores climaticos son mas favorables para salir a andar en bicicleta
'''

# Diagramas de pares

# sns.pairplot(
#     df
# )

# plt.show()

'''
**Preguntas:**

¿Existe alguna relación entre `temperature` y `dew_point_temperature`?
Si, ya que la temperatura de punto de rocio depende de la temperatura

De ser así, ¿de qué tipo y que significa para el problema de predicción de demanda de BikerPro?
Es una relacion lineal, lo cual significa que la temperatura de punto de rocio depende de la temperatura

Fijemos nuestra atención en la relación entre `hour` y `solar_radiation`. 
¿Que forma tiene el diagrama dispersión de ambos?
Una forma de linea recta

¿que explicación tiene la forma que muestran los valores de radicón según la hora?
Que la radiacion solar depende de la hora
'''

# Analisis Exploratorio Multivariado

# rented_bike_count vs hour vs seasons

# Especificamos el tamaño de la figura
# plt.figure(figsize=(20, 10))

# sns.lineplot(x='hour', y='rented_bike_count', hue='seasons', data=df).set(title='Cantidad de bicicletas rentadas por hora y por temporada')
# plt.show()

# plt.figure(figsize=(20, 10))

# sns.lineplot(x='hour', y='rented_bike_count', hue='holiday', data=df).set(title='Cantidad de bicicletas rentadas por hora y por temporada')
# plt.show()

'''
**Preguntas**

¿Qué tipo de comportamiento esperaria que tuviera el modelo en las predicciones por estación del año?
Que la demanda de servicios aumente en verano y disminuya en invierno

* Repita el análisis pero ahora cambiando `hue="holiday"`.
  * ¿Existe alguna diferencia en los patrones de renta dependiendo si es o no dia feriado?
   Si, ya que en los dias feriados la demanda de servicios es menor

  * ¿Los picos en la demanda el el caso de dias feriados son los mismos para días no feriados?
    No, ya que en los dias no feriados la demanda de servicios es mayor
'''

# rented_bike_count vs hour vs function_day

# Especificamos el tamaño de la figura
# plt.figure(figsize=(20, 10))

# sns.lineplot(x='hour', y='rented_bike_count', hue='functioning_day', data=df).set(title='Cantidad de bicicletas rentadas por hora y por tipo de dia')
# plt.show()

'''
**Preguntas**
* ¿Porqué en la figura anterio el segmento de `functioning_day=No` es una linea en cero?
Porque en los dias que no funcionan las bicicletas no se rentan

* Considerando lo anterior, 
¿es relevante incluir información de la variable `functioning_day` en el modelo predictivo?
Si, ya que la demanda de servicios depende de si las bicicletas funcionan o no

¿qué sucedería si no se incluyera?
Que el modelo no seria tan preciso
'''

# Diagramas para analisis multivariados

# Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# sns.scatterplot(x='temperature', y='rented_bike_count', hue='seasons', data=df).set(title='Temperatura vs Cantidad de bicicletas rentadas por temporada')
# plt.show()

# Replot

# sns.relplot(
#     x='temperature',
#     y='rented_bike_count',
#     hue='seasons',
#     col='seasons',
#     col_wrap=2,
#     data=df
# )

# plt.show()

# sns.relplot(
#     data=df,
#     x='temperature',
#     y='rented_bike_count',
#     hue='seasons',
#     col='hour',
#     col_wrap=3,
# )

# plt.show()

'''
**Preguntas**
¿Existe algun cambio en la tendencia de renta de bicletas y temperatura de acuerdo a la hora en que se segmente el análisis?
Si, ya que la demanda de servicios depende de la hora del dia y de la temperatura
'''

# FaceGrid

# Especificamos el tamaño de la figura
# plt.figure(figsize=(15, 6))

# g = sns.FacetGrid(df, col='seasons', hue='seasons')
# g.map(sns.scatterplot, "rented_bike_count", "temperature",)

# plt.show()

# Especifica el tamaño de la figura
# plt.figure(figsize = (15,6))

# g = sns.FacetGrid(df, col="seasons", row='holiday', hue='seasons')
# g.map(sns.scatterplot, "rented_bike_count", "temperature",)

# plt.show()

# Correlación

# Calcula la matriz de correlación
correlation = df.corr()

# Define el tamaño de la imagen
fig, ax = plt.subplots(figsize=(15, 15))

# Mapa de calor de la matriz de correlación
sns.heatmap(correlation, annot=True, linewidths=.5, ax=ax)

plt.show()

'''
**Preguntas**
¿Qué variables presenta una alta correlación positiva (mayor a 0.5) o negativa (menor a -0.5) contra rented_bike_count?
Las variables que presentan una alta correlacion positiva son: temperature, hour, solar_radiation, dew_point_temperature, 
humidity, visibility, ozone, pm10, pm2.5, y la variable que presenta una alta correlacion negativa es: wind_speed

¿`dew_point_temperature` tiene un alto grado de correlación positiva (mayor a 0.5) con cuales variables?
Con temperature

¿qué significa para este hecho problema?
Que la temperatura de punto de rocio depende de la temperatura

¿qué otros pares de variables presentan una alta correlación positiva (mayor a 0.5) o negativa (menor a -0.5)?
Las variables que presentan una alta correlacion positiva son: temperature y dew_point_temperature,

¿se deberían de incluir en el modelo variables que tengan una alta correlación entre si? ¿que consecuencias tendría incluirlas?
No, ya que el modelo no seria tan preciso
'''