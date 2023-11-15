from fastapi import FastAPI
from pydantic import BaseModel

class Scores(BaseModel):
    name: str
    math_score: float
    literature_score: float
    chemistry_score: float
    geography_score: float


app = FastAPI()

@app.post("/gpa/")
async def average_score(scores: Scores) -> dict:
    """
    Calcula el promedio de calificaciones del alumno
    """

    GPA = (scores.math_score+scores.literature_score+scores.chemistry_score+scores.geography_score)/4.0

    result = 'No Aprobado'
    if GPA > 6:
        result = 'Aprobado'

    return {"messages": f"Estudiante {scores.name} ha sido {result} con promedio {GPA}"}