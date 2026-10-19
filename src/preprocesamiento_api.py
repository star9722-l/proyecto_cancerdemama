from pydantic import BaseModel
from typing import List

class DatosCancerInput(BaseModel):
    # Usaremos los 10 datos con mayor correlación clínica (los "worst" son los más predictivos)
    caracteristicas: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "caracteristicas": [
                    0.2654,  # worst_concave_points
                    184.60,  # worst_perimeter
                    0.1471,  # mean_concave_points
                    25.38,   # worst_radius
                    122.80,  # mean_perimeter
                    0.7119,  # worst_concavity
                    2019.0,  # worst_area
                    17.99,   # mean_radius
                    0.4601,  # worst_symmetry
                    0.1189   # worst_fractal_dimension
                ]
            }
        }