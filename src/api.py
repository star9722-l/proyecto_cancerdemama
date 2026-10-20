from fastapi import FastAPI, HTTPException
from src.preprocesamiento_api import DatosCancerInput
import joblib
import numpy as np
import os
from tensorflow import keras

app = FastAPI(title="API Cáncer de Mama - Grupo 03")

# Cargar modelo y escalador
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
        # Creamos un arreglo de 30 ceros y ponemos los 10 datos en las posiciones correctas
        input_completo = np.zeros((1, 30))
        
        # Estos son los índices de las 10 variables que usamos en el chatbot
        indices = [22, 25, 7, 20, 2, 26, 23, 0, 28, 29] 
        input_completo[0, indices] = datos.caracteristicas
        
        # Escalar y predecir
        input_scaled = scaler.transform(input_completo)
        prob = float(modelo.predict(input_scaled)[0][0])
        
        # 0 = Maligno, 1 = Benigno
        clase = "Maligno" if prob < 0.5 else "Benigno"
        
        return {
            "prediccion": clase,
            "probabilidad_benigno": round(prob, 4),
            "advertencia": "Este resultado es una alerta y requiere confirmación médica."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")