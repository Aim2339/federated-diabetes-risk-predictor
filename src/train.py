from preprocessing import preprocess_data
from model import model

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save("../results/mlp_model.keras")

# Test the model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)