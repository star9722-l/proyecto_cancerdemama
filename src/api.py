from fastapi import FastAPI, HTTPException
from src.preprocesamiento_api import DatosCancerInput
import joblib
import numpy as np
import os
from tensorflow import keras

app = FastAPI(title="API Cáncer de Mama")

# Cargar modelo
modelo = keras.models.load_model("modelos/modelo_promovido.keras")
scaler = joblib.load("modelos/scaler.pkl")

@app.get("/salud")
def salud():
    return {"status": "ok", "mensaje": "API funcionando"}

@app.post("/predecir")
def predecir(datos: DatosCancerInput):
    if len(datos.caracteristicas) != 10:
        raise HTTPException(status_code=400, detail="Se requieren exactamente 10 características.")
    
    try:
        # Rellenamos un arreglo de 30 ceros y ponemos los 10 datos del usuario 
        # en las posiciones que corresponden para no romper el modelo
        input_completo = np.zeros((1, 30))
        
        # Asumiendo que el orden es el mismo que pusimos en el chatbot
        indices = [22, 25, 7, 20, 2, 26, 23, 0, 28, 29] 
        input_completo[0, indices] = datos.caracteristicas
        
        input_scaled = scaler.transform(input_completo)
        
        prob = float(modelo.predict(input_scaled)[0][0])
        clase = "Maligno" if prob < 0.5 else "Benigno"
        
        return {
            "prediccion": clase,
            "probabilidad_benigno": round(prob, 4),
            "advertencia": "Este resultado requiere confirmación médica."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")