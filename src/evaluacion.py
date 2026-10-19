from fastapi.testclient import TestClient
from src.api import app

cliente = TestClient(app)

def test_endpoint_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"

def test_predecir_datos_invalidos():
    # Enviamos solo 5 datos en lugar de 30
    respuesta = cliente.post("/predecir", json={"caracteristicas": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert respuesta.status_code == 400

def test_predecir_datos_validos():
    # 30 datos de ejemplo
    datos_dummy = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                   1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                   25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    respuesta = cliente.post("/predecir", json={"caracteristicas": datos_dummy})
    assert respuesta.status_code == 200
    assert "prediccion" in respuesta.json()