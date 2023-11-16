"""
Ejemplo 4: Draft de la interfaz grafica para CueBot
Sin embargo falta la conexion a ChatGPT :(
"""


import gradio as gr
from PyPDF2 import PdfReader
import random
import time


# Variable auxiliar para guardar el texto del PDF
CORPUS_TEXT = ''

# Respuestas aleatorias para prueba del chatbot
RESPUESTAS_ALEATORIAS = [
    "Soy la respuesta aleatoria 1...",
    "Soy la respuesta aleatoria 2...",
    CORPUS_TEXT
]


def add_text(history, text):
    """
    Agrega texto a la historia del chat y actualiza la
    interfaz
    """

    history = history + [(text, None)]

    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    """
    Permite agrega un texto pdf a la conversacion del chat
    y guardar 
    """
    # agrega el nombre del archivo al Chat
    history = history + [((file.name,), None)]

    # Leemos el texto del archivo PDF y lo guardamos en
    # CORPUS_TEXT para el futuro

    # Abre el archivo PDF en modo lectura binaria
    with open(file.name, 'rb') as pdf_file:
        # Crea un lector de PDF
        pdf_reader = PdfReader(pdf_file)

        # Inicializa una variable para almacenar el texto extraído
        extracted_text = ""

        # Itera sobre todas las páginas del PDF y extrae el texto
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page_num].extract_text()

        # Comunica el texto recien leido a la variable global CORPUS_TEXT
        global CORPUS_TEXT
        CORPUS_TEXT = extracted_text
        print(CORPUS_TEXT)
    return history


def bot(history):
    """
    Obtiene la respuesta del Bot
    """
    # Genera una respuesta aleatoria para hacer prueba
    response = random.choice(RESPUESTAS_ALEATORIAS)

    # Extrae el ultimo input de texto de la historia de la 
    # conversacion del bot
    input_text = history[-1][0]  
    if "cuento" in input_text:
        response = "Tienes que modificar esta parte para hablar con ChatGPT"
    else:
        response = response
    
    # Define entrada de texto vacio
    history[-1][1] = ""

    # Genera el efecto de escribir lento con una pausa
    # Como si el texto se generara lentamente
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history



# Crea la aplicacion de Gradio
with gr.Blocks() as demo:

    # Crea ek chatbot
    chatbot = gr.Chatbot([], elem_id="chatbot", height=750)


    with gr.Row():

        # Cuadro de texto de conversacion
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Especifica el archivo pdf o ingresa un texto",
                container=False
            )

        # Cuadro de subida de archivo
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁 Subir Archivo:", file_types=["pdf"])

    # Controlador de acciones para retroalimentar al bot
    # Con su respuesta
    txt_msg = txt.submit(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt],
        queue=False
        ).success(
        fn=bot,
        inputs=chatbot,
        outputs=chatbot
    )

    # Actualiza la convesacion con el texto generado
    txt_msg.success(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=[txt],
        queue=False
        )

    # Sube el archivo  y actualiza la conversacion
    file_msg = btn.upload(
        fn=add_file,
        inputs=[chatbot, btn],
        outputs=[chatbot],
        queue=False).success(
        bot, chatbot, chatbot
    )


demo.queue()
if __name__ == "__main__":
    demo.launch(
    share=True
    )
