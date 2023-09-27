# Ejemplo 1 de una aplicacion en Flask
from flask import Flask

# Instancia el app
app = Flask(__name__)

# Ruta raiz
@app.route('/')
def hello_world():
    return 'Hola Mundo! Soy tu primer ejemplo ;)'

if __name__ == '__main__':
    app.run()