from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression

def crear_baseline():
    return LogisticRegression(max_iter=1000, random_state=42)

def crear_red_simple(input_shape):
    modelo = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.Recall(name="recall")])
    return modelo

def crear_red_con_dropout(input_shape):
    modelo = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.Recall(name="recall")])
    return modelo