import sys, os, json, datetime
from sklearn.metrics import accuracy_score, recall_score, f1_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from datos import cargar_y_particionar
from modelo import crear_baseline, crear_red_simple, crear_red_con_dropout

def evaluar(nombre, y_real, y_pred):
    return {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "recall": float(recall_score(y_real, y_pred)),
        "f1": float(f1_score(y_real, y_pred))
    }

def main():
    print("Cargando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_y_particionar()

    print("Entrenando Baseline...")
    modelo_base = crear_baseline()
    modelo_base.fit(X_train, y_train)
    res_base = evaluar("Regresión Logística", y_val, modelo_base.predict(X_val))

    print("Entrenando Red Simple...")
    red_simple = crear_red_simple(X_train.shape[1])
    red_simple.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    res_simple = evaluar("Red Simple", y_val, (red_simple.predict(X_val) > 0.5).astype(int).flatten())

    print("Entrenando Red con Dropout...")
    red_dropout = crear_red_con_dropout(X_train.shape[1])
    red_dropout.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    res_dropout = evaluar("Red con Dropout", y_val, (red_dropout.predict(X_val) > 0.5).astype(int).flatten())

    # Elegir ganador en Validación (Priorizando Recall)
    resultados = [res_base, res_simple, res_dropout]
    ganador = max(resultados, key=lambda x: x["recall"])
    print(f"Ganador (Val): {ganador['modelo']}")

    # Evaluar en Test
    if ganador["modelo"] == "Regresión Logística":
        pred_test = modelo_base.predict(X_test)
        modelo_final, formato = modelo_base, "pkl"
    elif ganador["modelo"] == "Red Simple":
        pred_test = (red_simple.predict(X_test) > 0.5).astype(int).flatten()
        modelo_final, formato = red_simple, "keras"
    else:
        pred_test = (red_dropout.predict(X_test) > 0.5).astype(int).flatten()
        modelo_final, formato = red_dropout, "keras"

    res_test = evaluar(ganador["modelo"], y_test, pred_test)
    print(f"Resultados Test: {res_test}")

    # Guardar modelo y ficha
    if formato == "pkl": joblib.dump(modelo_final, "modelos/modelo_promovido.pkl")
    else: modelo_final.save("modelos/modelo_promovido.keras")

    ficha = {
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_ganador": ganador["modelo"],
        "metricas_val": ganador,
        "metricas_test": res_test
    }
    with open("modelos/ficha_modelo.json", "w") as f:
        json.dump(ficha, f, indent=4)
    print("¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()