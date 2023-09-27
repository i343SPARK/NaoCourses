"""
Aplicacion que crea un API REST para obtener
las prediccion de una regresion lineal 
previamente entrenada
"""

# Flask
from flask import Flask, request
import pickle
import os

# --------- Aplicacion en Flask ------------------- #
# Instancia la app
app = Flask(__name__, template_folder='.')

# Define una ruta para saber si la aplicacion 
# esta corriendo
@app.route('/healtCheck')
def index():
    return "true"

# Ruta con accion get para obtener la predicion
@app.route('/prediction/', methods=['GET'])
def get_prediction():
    # Convierte el parametro f recibido por GET a
    # numero flotante
    feature = float(request.args.get('feature'))

    # Cargamos el modelo ya entrenado
    CURRENT_PATH = app.static_folder.replace('static', '')
    model_path = os.path.join(CURRENT_PATH, 'model_diabetes.pkl')
    model = pickle.load(open(model_path, 'rb'))
    # Obtiene la prediccion para el valor recibido
    pred = model.predict([[feature]])

    # Comunimos a la ruta el valor de la prediccion
    return str(pred)


if __name__ == '__main__':
    app.run()
