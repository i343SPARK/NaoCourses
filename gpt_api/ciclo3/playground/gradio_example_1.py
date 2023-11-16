"""
Ejemplo 1: Aplicacion que crea una version al reves de
un texto
"""

import gradio as gr
import time


def reversing_word(palabra):
    """
    Obtiene una version al reves de una palabra
    """
    time.sleep(1)

    new_string = ""

    for letra in palabra:
        new_string = letra + new_string

    return new_string


# Interfaz de la aplicacion
demo = gr.Interface(
    fn=reversing_word,
    inputs=gr.Text(),
    outputs=gr.Text()
    )

# Despliegue de aplicacion
demo.launch()
