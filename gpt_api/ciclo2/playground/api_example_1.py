"""
Implementacion de API Rest con FastAPI

Ejemplo 1
"""

from fastapi import FastAPI

# Crea una aplicacion de FastAPI
app = FastAPI()

# Define la ruta raiz (root) que devuelve un diccionario
# junto con un mensaje de bievenida
@app.get("/")
async def root():
    """
    Ruta raiz
    """
    return {"mensaje": "Hola :)! Soy tu primer ejemplo en FastAPI"}
