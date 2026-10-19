import requests

def iniciar_chatbot():
    print("🤖 Hola. Soy tu asistente de apoyo para laboratorio clínico.")
    print("Te ayudaré a estimar el riesgo basado en 30 medidas celulares.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        print("Por favor, ingresa las 30 medidas separadas por comas:")
        entrada = input("> ")
        
        if entrada.lower() == 'salir':
            print("🤖 Recuerda que este sistema no reemplaza a un médico. ¡Hasta luego!")
            break
            
        try:
            # Convertir entrada a lista de números
            valores = [float(x.strip()) for x in entrada.split(",")]
            
            if len(valores) != 30:
                print(f"⚠️ Error: Ingresaste {len(valores)} valores. Necesito exactamente 30.\n")
                continue
                
            # Consultar API
            respuesta = requests.post("http://127.0.0.1:8000/predecir", json={"caracteristicas": valores})
            datos = respuesta.json()
            
            if respuesta.status_code == 200:
                print(f"\n🔬 Resultado: {datos['prediccion']}")
                print(f"Probabilidad de benignidad: {datos['probabilidad_benigno'] * 100}%")
                print(f"⚠️ {datos['advertencia']}\n")
                print("¿Deseas derivar este caso a un especialista humano? (si/no)")
                derivar = input("> ")
                if derivar.lower() == 'si':
                    print("🤖 Entendido. El caso ha sido marcado para revisión médica prioritaria.\n")
                else:
                    print("🤖 Entendido. Puedes ingresar otro caso.\n")
            else:
                print(f"⚠️ Error en la API: {datos.get('detail')}\n")
                
        except ValueError:
            print("⚠️ Error: Asegúrate de ingresar solo números separados por comas.\n")
        except requests.exceptions.ConnectionError:
            print("⚠️ Error: No se pudo conectar con la API. ¿Ya la iniciaste?\n")
            break

if __name__ == "__main__":
    iniciar_chatbot()