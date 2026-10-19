from fastapi import FastAPI, HTTPException
from src.preprocesamiento_api import DatosCancerInput
import joblib
import numpy as np
import os
from tensorflow import keras

app = FastAPI(title="API Cáncer de Mama")

# Cargar modelo al arrancar (Eficiencia)
RUTA_MODELO_KERAS = "modelos/modelo_promovido.keras"
RUTA_MODELO_PKL = "modelos/modelo_promovido.pkl"
scaler = joblib.load("modelos/scaler.pkl")

if os.path.exists(RUTA_MODELO_KERAS):
    modelo = keras.models.load_model(RUTA_MODELO_KERAS)
    es_keras = True
elif os.path.exists(RUTA_MODELO_PKL):
    modelo = joblib.load(RUTA_MODELO_PKL)
    es_keras = False
else:
    raise Exception("No se encontró el modelo entrenado.")

@app.get("/salud")
def salud():
    return {"status": "ok", "mensaje": "API funcionando"}

@app.post("/predecir")
def predecir(datos: DatosCancerInput):
    if len(datos.caracteristicas) != 30:
        raise HTTPException(status_code=400, detail="Se requieren exactamente 30 características.")
    
    try:
        # Preparar datos
        input_array = np.array(datos.caracteristicas).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Predecir
        if es_keras:
            prob = float(modelo.predict(input_scaled)[0][0])
        else:
            prob = float(modelo.predict_proba(input_scaled)[0][1])
            
        clase = "Maligno" if prob < 0.5 else "Benigno" # 0=Maligno, 1=Benigno
        
        return {
            "prediccion": clase,
            "probabilidad_benigno": round(prob, 4),
            "advertencia": "Este resultado es una alerta y requiere confirmación médica."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")