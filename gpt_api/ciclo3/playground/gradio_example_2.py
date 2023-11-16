"""
Ejemplo 2: Aplicacion que crea un ChatBot sencillo
que da una respuesta personalizada segun el mensaje
que reciba.
"""

import gradio as gr


def echo(message, history):
    """
    Devuelve una respuesta personalizada segun
    el mensaje
    """
    if message.lower() == "hello":
        return "word"
    elif message.lower() == "hola":
        return "mundo"
    else:
        return "Hay una serpiente en mi bota"


# Interfaz de la aplicacion del chatbot
demo = gr.ChatInterface(
    # funcion de procesamiento de respuesta
    fn=echo,
    # Ayuda a mostrar respuestas de ejemplo
    examples=["hello", "hola"],
    # Titulo de la interfaz
    title="Mi primer ChatBot en Gradio!!!"
    )

# Despliga el app
demo.launch()
