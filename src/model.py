from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Create the MLP model
model = Sequential([
    Input(shape=(21,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Show model structure
model.summary()