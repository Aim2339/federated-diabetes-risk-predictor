from preprocessing import load_and_preprocess

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load the clients
clients = load_and_preprocess()

# Sigmoid function
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

# Local Logistic Regression training
def train_local_model(X, y, weights, bias):

    learning_rate = 0.01
    local_epochs = 20

    for epoch in range(local_epochs):
        predictions = sigmoid(
            np.dot(X, weights) + bias
        )

        errors = predictions - y

        weight_gradient = np.dot(
            X.T,
            errors
        ) / len(X)

        bias_gradient = np.sum(errors) / len(X)
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient
    return weights, bias

# Initialize global model
global_weights = np.zeros(4)
global_bias = 0.0

# Federated Training
number_of_rounds = 10

for round_number in range(number_of_rounds):

    print("\nFEDERATED ROUND", round_number + 1)

    client_weights = []
    client_biases = []
    client_sizes = []

    # Train each client
    for client in clients:

        print("\nTraining:", client["name"])

        local_weights = global_weights.copy()
        local_bias = global_bias

        local_weights, local_bias = train_local_model(
            client["X_train"],
            client["y_train"],
            local_weights,
            local_bias
        )

        client_weights.append(local_weights)
        client_biases.append(local_bias)
        client_sizes.append(len(client["X_train"]))

    # Weighted Federated Averaging
    total_samples = sum(client_sizes)

    print("\nTotal training samples:", total_samples)

    global_weights = np.zeros(4)
    global_bias = 0.0

    for i in range(len(clients)):

        client_weight = client_sizes[i] / total_samples

        global_weights += (
            client_weight * client_weights[i]
        )

        global_bias += (
            client_weight * client_biases[i]
        )

    print("Global model updated.")

# Final Global Model Evaluation
print("\nFINAL GLOBAL MODEL")
print("Using threshold: 0.5")

all_actual_values = []
all_predictions = []
all_probabilities = []


# Test the global model on each client
for client in clients:

    name = client["name"]

    X_test = client["X_test"]
    y_test = client["y_test"]

    probabilities = sigmoid(
        np.dot(X_test, global_weights) + global_bias
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    all_actual_values.extend(y_test)
    all_predictions.extend(predictions)
    all_probabilities.extend(probabilities)

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # Display results

    print("\n" + name)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix:")
    print(matrix)

# Overall Results
print("\nOVERALL FINAL GLOBAL MODEL")

overall_accuracy = accuracy_score(
    all_actual_values,
    all_predictions
)

overall_precision = precision_score(
    all_actual_values,
    all_predictions
)

overall_recall = recall_score(
    all_actual_values,
    all_predictions
)

overall_f1 = f1_score(
    all_actual_values,
    all_predictions
)

overall_roc_auc = roc_auc_score(
    all_actual_values,
    all_probabilities
)

overall_matrix = confusion_matrix(
    all_actual_values,
    all_predictions
)


print("\nAccuracy:", overall_accuracy)
print("Precision:", overall_precision)
print("Recall:", overall_recall)
print("F1-Score:", overall_f1)
print("ROC-AUC:", overall_roc_auc)
print("\nOverall Confusion Matrix:")
print(overall_matrix)