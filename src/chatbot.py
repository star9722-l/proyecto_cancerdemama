import requests

# Los 10 datos que elegimos (basados en tu imagen de correlación)
PREGUNTAS = [
    ("worst_concave_points", 0.0, 0.3),     # Rango típico: 0 a 0.3
    ("worst_perimeter", 50.0, 300.0),       # Rango típico: 50 a 300
    ("mean_concave_points", 0.0, 0.2),      # Rango típico: 0 a 0.2
    ("worst_radius", 5.0, 40.0),            # Rango típico: 5 a 40
    ("mean_perimeter", 40.0, 200.0),        # Rango típico: 40 a 200
    ("worst_concavity", 0.0, 1.5),          # Rango típico: 0 a 1.5
    ("worst_area", 150.0, 4000.0),          # Rango típico: 150 a 4000
    ("mean_radius", 5.0, 30.0),             # Rango típico: 5 a 30
    ("worst_symmetry", 0.1, 0.7),           # Rango típico: 0.1 a 0.7
    ("worst_fractal_dimension", 0.04, 0.2)  # Rango típico: 0.04 a 0.2
]

def iniciar_chatbot():
    print("=====================================================")
    print("🤖 Hola. Soy tu asistente clínico virtual.")
    print("Te haré 10 preguntas para evaluar el tumor.")
    print("Escribe 'salir' en cualquier momento para terminar.")
    print("=====================================================\n")

    valores = []
    
    for nombre, minimo, maximo in PREGUNTAS:
        while True:
            # Si el usuario se equivoca, el bucle 'while' se repite para esta misma pregunta
            entrada = input(f"👉 Ingrese el {nombre}: ")
            
            if entrada.lower() == 'salir':
                print("🤖 ¡Hasta luego!")
                return

            try:
                valor = float(entrada)
                
                # AQUÍ ESTÁ LA VALIDACIÓN DE ERRORES QUE PIDE EL INGENIERO
                if valor < minimo or valor > maximo:
                    print(f"⚠️ ERROR: El valor ingresado es anormal. Para {nombre}, el rango clínico aceptado es entre {minimo} y {maximo}.")
                    print("Volvamos a intentarlo.\n")
                    continue # Vuelve a preguntar lo mismo
                
                valores.append(valor)
                break # Si el dato es correcto, pasa a la siguiente pregunta
                
            except ValueError:
                print("⚠️ ERROR: Debes ingresar un número válido (usa punto para decimales).\n")

    print("\n🔄 Consultando a la API y al modelo de Inteligencia Artificial...")
    
    try:
        respuesta = requests.post("http://127.0.0.1:8000/predecir", json={"caracteristicas": valores})
        datos = respuesta.json()
        
        if respuesta.status_code == 200:
            print("\n✅ --- RESULTADO DEL ANÁLISIS ---")
            print(f"🔬 Diagnóstico probable: {datos['prediccion']}")
            print(f"📊 Probabilidad de benignidad: {datos['probabilidad_benigno'] * 100:.2f}%")
            print(f"⚠️ {datos['advertencia']}")
            print("-------------------------------\n")
            
            derivar = input("¿Deseas derivar este caso a un especialista humano? (si/no): ")
            if derivar.lower() == 'si':
                print("\n🏥 Caso marcado para revisión médica prioritaria.\n")
            else:
                print("\n🤖 Entendido. Fin de la consulta.\n")
                
        else:
            print(f"\n⚠️ Error en la API: {datos.get('detail')}\n")
            
    except requests.exceptions.ConnectionError:
        print("\n⚠️ ERROR CRÍTICO: No se pudo conectar con la API. ¿Ya encendiste el servidor (uvicorn)?\n")

if __name__ == "__main__":
    iniciar_chatbot()