"""
Ejemplo 3: Aplicacion que crea una interfaz para
subir archivo
"""
import gradio as gr


def upload_file(files):
    """
    Funcion de procesamiento de archivos
    subidos a la aplicación
    """

    # Procesamiento dummy de los archivos
    print("Subiendo archivo ...")
    file_paths = [file.name for file in files]

    for file in files:
        print("Ubicacion del archivo subido: ", file.name)

    return file_paths


# Crea el diseño de la app en bloques de componentes
with gr.Blocks() as demo:

    # Crea componente tipo archivo
    file_output = gr.File()

    # Crean boton accionable para subir archivo
    upload_button = gr.UploadButton(
        "Click to Upload a File",
        file_types=["image", "video", "pdf"],
        file_count="multiple"
        )

    # realiza la subida del archivo y su procesamiento
    upload_button.upload(
        upload_file,
        upload_button,
        file_output
        )

# Despliga el app
demo.launch(debug=True)