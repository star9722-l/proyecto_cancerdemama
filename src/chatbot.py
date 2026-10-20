import requests

# Definimos las 10 variables (empezando por las más predictivas) con sus rangos REALES
VARIABLES = [
    {"nombre": "worst concave points", "min": 0.0, "max": 0.291},
    {"nombre": "worst perimeter", "min": 50.41, "max": 251.20},
    {"nombre": "mean concave points", "min": 0.0, "max": 0.201},
    {"nombre": "worst radius", "min": 7.93, "max": 36.04},
    {"nombre": "mean perimeter", "min": 43.79, "max": 188.50},
    {"nombre": "worst concavity", "min": 0.0, "max": 1.252},
    {"nombre": "worst area", "min": 185.2, "max": 4254.0},
    {"nombre": "mean radius", "min": 6.981, "max": 28.11},
    {"nombre": "worst symmetry", "min": 0.1565, "max": 0.6638},
    {"nombre": "worst fractal dimension", "min": 0.05504, "max": 0.2075}
]

def iniciar_chatbot():
    print("==================================================================")
    print("🤖 Hola, soy tu asistente clínico virtual para el Cáncer de Mama.")
    print("Te haré 10 preguntas clave para estimar el riesgo del tumor.")
    print("Escribe 'salir' en cualquier momento para cancelar.")
    print("==================================================================\n")

    datos_ingresados = []

    # 1. RECOLECCIÓN DE DATOS PASO A PASO (SLOTS) CON VALIDACIÓN
    for variable in VARIABLES:
        nombre = variable["nombre"]
        minimo = variable["min"]
        maximo = variable["max"]
        
        while True: # Bucle para que insista si el usuario se equivoca
            entrada = input(f"👉 Ingrese el valor de {nombre}: ")
            
            if entrada.lower() == 'salir':
                print("\n🤖 Consulta cancelada. ¡Hasta luego!")
                return
            
            try:
                valor = float(entrada)
                
                # VALIDACIÓN DE OUTLIERS REALES
                if valor < minimo or valor > maximo:
                    print(f"   ⚠️ ERROR: Dato clínico fuera de rango. Para '{nombre}' el valor real debe estar entre {minimo} y {maximo}.")
                    print("   Por favor, vuelva a intentarlo.\n")
                    continue # Vuelve a pedir el mismo dato
                
                datos_ingresados.append(valor)
                break # Si el dato es correcto, pasa a la siguiente variable
                
            except ValueError:
                print("   ⚠️ ERROR: Formato inválido. Ingrese un número (use punto para decimales).\n")

    # 2. CONSULTA A LA API
    print("\n🔄 Analizando los datos con la Inteligencia Artificial...")
    
    try:
        # Hacemos la petición a la API en local
        respuesta = requests.post(
            "http://127.0.0.1:8000/predecir", 
            json={"caracteristicas": datos_ingresados},
            timeout=5
        )
        
        # 3. MANEJO DE RESPUESTAS Y DERIVACIÓN HUMANA
        if respuesta.status_code == 200:
            datos = respuesta.json()
            
            print("\n✅ --- RESULTADO DEL ANÁLISIS CLÍNICO ---")
            print(f"🔬 Clasificación del modelo: {datos['prediccion'].upper()}")
            print(f"📊 Probabilidad de benignidad: {datos['probabilidad_benigno'] * 100:.2f}%")
            print(f"⚠️  Advertencia: {datos['advertencia']}")
            print("------------------------------------------\n")
            
            # Flujo alternativo: Derivación a humano
            derivar = input("❓ ¿Deseas derivar este caso a un especialista humano? (si/no): ")
            if derivar.lower() == 'si':
                print("\n🏥 Caso marcado para revisión médica prioritaria. Un especialista se contactará.\n")
            else:
                print("\n🤖 Entendido. Recuerde que este sistema es una herramienta de apoyo.\n")
                
        else:
            # Si la API devuelve error 400 o 500
            error_msg = respuesta.json().get("detail", "Error desconocido en el servidor")
            print(f"\n⚠️ Error de validación en la API: {error_msg}\n")
            
    except requests.exceptions.ConnectionError:
        print("\n🚨 ERROR CRÍTICO: No se pudo conectar con la API.")
        print("Asegúrese de haber encendido el servidor con: uvicorn src.api:app --reload\n")
    except Exception as e:
        print(f"\n🚨 Ocurrió un error inesperado: {e}\n")

if __name__ == "__main__":
    iniciar_chatbot()