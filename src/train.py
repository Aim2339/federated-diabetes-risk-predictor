from preprocessing import preprocess_data
from model import create_model

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

# Create model
model = create_model()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save("../results/baseline_model.keras")

# Test the model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)