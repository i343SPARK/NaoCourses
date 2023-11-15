"""
Implementacion de API Rest con FastAPI

Ejemplo 2
"""

from fastapi import FastAPI


# Crea una aplicacion de FastAPI
app = FastAPI()

# Ruta de saludo (estatica)
@app.get("/student/grettings")
async def read_grettings() -> str:
    """
    Ruta de saludos
    """
    return "Bienvenido estudiante!!"

# Ruta de saludo (dinamica)
@app.get("/student/{student_id}")
async def read_student_id(student_id: int) -> str:
    return f"Bienvenido estudiante numero {student_id}"

# Ruta de disponibilidad de catalogos de libros
@app.get("/books/availability")
async def read_book_availability(year: str = '2023', month: str = '1') -> dict:
    """
    Ruta que indica la disponibilidad de catalogos de libros
    por mes y año
    """

    # Diccionario de disponibilidad de catalogos de libros
    books = {
        '2022': {
            '1': 'Not Available',
            '2': 'Available',
            '3': 'Available',
            '4': 'Not Available',
            '5': 'Available',
            '6': 'Not Available',
            '7': 'Available',
            '8': 'Available',
            '9': 'Available',
            '10': 'Available',
            '11': 'Available',
            '12': 'Available',
        },
        '2023': {
            '1': 'Not Available',
            '2': 'Available',
            '3': 'Available',
            '4': 'Not Available',
            '5': 'Available',
            '6': 'Not Available',
            '7': 'Available',
            '8': 'Not Available',
            '9': 'Not Available',
            '10': 'Not Available',
            '11': 'Not Available',
            '12': ' NotAvailable',
        },
    }

    # Consulta el estatus por año y mes
    status = books[year][month]

    return {"messages": f"Catalog for {year}-{month} is {status}"}
