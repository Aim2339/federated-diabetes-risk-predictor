from preprocessing import preprocess_data
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Get the test data
X_train, X_test, y_train, y_test = preprocess_data()

# Load the trained model
model = load_model("../results/federated_model.keras")

# Get predictions
probabilities = model.predict(X_test)
predictions = (probabilities >= 0.4).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
matrix = confusion_matrix(y_test, predictions)

# Display results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")
print(matrix)