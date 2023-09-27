"""
Entrena una regresion lineal sobre el conjunto
diabetes de Skleanr
"""

import pickle
import numpy as np

# Sklearn
from sklearn import linear_model
from sklearn import datasets

# Conjunto de datos diabetes
diabetes = datasets.load_diabetes()

# Entrenamos el modelo de regresion lineal con
# una sola variable
X = diabetes.data[:, np.newaxis, 2]
model = linear_model.LinearRegression()
model.fit(X, diabetes.target)

# Salvamos el modelo entrenado con formato pickle
pickle.dump(model, open('model_diabetes.pkl', 'wb'))

