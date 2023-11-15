import os
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/v1/upload")
async def upload_file(file: UploadFile):
    """
    Ayuda a subir un archivo desde POST
    """
    # Obtiene el nombre del archivo
    filename = file.filename

    # Lee el contenido del archivo
    file_content = await file.read()

    # Se asegura que exista la carpeta /tmp
    # en el directorio de trabajo
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    # escribe una 
    with open('./tmp/' + "copy_"+filename, "wb") as f:
        f.write(file_content)

    return {"message": "Copia del archivo subida en folder ./tmp"}