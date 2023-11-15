import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

"""
# Ejecuta una tarea de completar un texto
chat_completion = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "system",
            "content": "Actúa como experto en redacciones para marketing digital."
        },
        {
            "role": "user",
            "content": "Ayudame a crear 3 propuestas de slogan para un equipo de baseball que tiene a un zorro como mascota en 5 palabras"
        }
    ]
)
# Imprime el mensaje respuesta de ChatGPT
print(chat_completion.choices[0].message.content)
"""

"""
# Ejecuta una tarea de completar un texto
completion = openai.completions.create(
    model = "text-davinci-003",
    prompt = "Roses are red, Violets are blue",
    max_tokens = 7,
    temperature = 0
)

print(completion.choices[0].text)
"""

### Extraccion de texto de un archivo .txt ###

def reading_txt(file_path: str) -> str:
    """
    Extrae el texto de un archivo .txt
    
    Parameters;
        file_path (str): Ruta del archivo .txt a analizar

    Salida:
        str: El texto extraído del archivo.
    """

    with open(file_path, "r") as file:
        return file.read()
    
# Ruta del archivo .txt a analizar
txt_path_file = "./tale_of_two_cities_chapter_1.txt"

# Extrae el texto del archivo .txt
extracted_text = reading_txt(txt_path_file)

# Imprime el texto extraído
# print(extracted_text)

### Extraccion de texto de un archivo .pdf ###
from PyPDF2 import PdfReader

def reading_pdf(file_path: str) -> str:
    """
    Extrae el texto de un archivo .pdf
    
    Parameters;
        file_path (str): Ruta del archivo .pdf a analizar

    Salida:
        str: El texto extraído del archivo.
    """

    reader = PdfReader(file_path)
    page = reader.pages[0]
    return page.extract_text()

# Ruta del archivo .pdf a analizar
pdf_path_file = "./cuento.pdf"

# Extrae el texto del archivo .pdf
extracted_pdf = reading_pdf(pdf_path_file)

# Imprime el texto extraído
# print(extracted_pdf)