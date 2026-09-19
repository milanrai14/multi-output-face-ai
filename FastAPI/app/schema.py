from pydantic import BaseModel


class PredictionResponse(BaseModel):
    age: int
    gender: str
    gender_probability: float
