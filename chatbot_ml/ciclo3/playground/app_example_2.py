"""
Ejemplo 2 de una aplicacion en Flask
Renderizando un archivo de HTML
"""

# Importamos "render_template"
from flask import Flask, render_template

# Instancia el app
# Nota: Cuida que example_1.html este
# en el mismo directorio donde esta este archivo!
app = Flask(__name__, template_folder='.')

# Definemos la ruta para 
@app.route("/")
def render():
    # Indicamos que la ruta de regresar el template 
    # renderizado del archivo "example_1.html"
    return render_template("index.html")

# Iniciamos la aplicación
if __name__ == '__main__':
    app.run()