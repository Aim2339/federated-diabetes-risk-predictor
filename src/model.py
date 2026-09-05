from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_model():
    model = Sequential([
        Input(shape=(21,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model