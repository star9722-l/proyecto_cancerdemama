import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def cargar_y_particionar():
    ruta_archivo = os.path.join("data", "wdbc.data")
    df = pd.read_csv(ruta_archivo, header=None)

    base_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
    feature_names = []
    for prefix in ["mean_", "se_", "worst_"]:
        for name in base_names:
            feature_names.append(f"{prefix}{name}")
            
    df.columns = ["id", "diagnosis"] + feature_names
    df["target"] = df["diagnosis"].map({"M": 0, "B": 1})
    
    X = df[feature_names]
    y = df["target"]

    # 1. Dividir (70% Train, 15% Val, 15% Test) SIN FUGA
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Imputar nulos (fit solo en train)
    imputador = SimpleImputer(strategy="median")
    X_train = imputador.fit_transform(X_train)
    X_val = imputador.transform(X_val)
    X_test = imputador.transform(X_test)

    # 3. Escalar (fit solo en train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Guardar preprocesadores para la API
    os.makedirs("modelos", exist_ok=True)
    joblib.dump(scaler, "modelos/scaler.pkl")
    joblib.dump(imputador, "modelos/imputador.pkl")
    joblib.dump(feature_names, "modelos/feature_names.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test