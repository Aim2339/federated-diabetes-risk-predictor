from preprocessing import load_and_preprocess

import numpy as np
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# Load the three clients

clients = load_and_preprocess()


# Logistic Regression functions

def sigmoid(value):

    return 1 / (1 + np.exp(-value))


def train_local_model(X, y, weights, bias):

    learning_rate = 0.01
    local_epochs = 20

    for epoch in range(local_epochs):

        # Calculate prediction

        linear_output = np.dot(X, weights) + bias

        predictions = sigmoid(linear_output)

        # Calculate errors

        errors = predictions - y

        # Calculate gradients

        weight_gradient = np.dot(X.T, errors) / len(X)

        bias_gradient = np.sum(errors) / len(X)

        # Update weights

        weights = weights - learning_rate * weight_gradient

        # Update bias

        bias = bias - learning_rate * bias_gradient

    return weights, bias


# Initialize global model

number_of_features = 4

global_weights = np.zeros(number_of_features)
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

        name = client["name"]

        X_train = client["X_train"]
        y_train = client["y_train"]

        print("\nTraining:", name)

        # Start from the current global model

        local_weights = global_weights.copy()
        local_bias = global_bias

        # Train locally

        local_weights, local_bias = train_local_model(
            X_train,
            y_train,
            local_weights,
            local_bias
        )

        # Store local parameters

        client_weights.append(local_weights)
        client_biases.append(local_bias)

        # Store number of training samples

        client_sizes.append(len(X_train))


    # Weighted Federated Averaging

    total_samples = sum(client_sizes)

    print("\nTotal training samples:", total_samples)

    new_global_weights = np.zeros(number_of_features)
    new_global_bias = 0.0

    for i in range(len(clients)):

        client_weight = client_sizes[i] / total_samples

        new_global_weights = (
            new_global_weights
            + client_weight * client_weights[i]
        )

        new_global_bias = (
            new_global_bias
            + client_weight * client_biases[i]
        )


    # Update global model

    global_weights = new_global_weights
    global_bias = new_global_bias

    print("Global model updated.")


# Save the final global model

results_folder = "../results"

os.makedirs(results_folder, exist_ok=True)

np.save(
    "../results/global_weights.npy",
    global_weights
)

np.save(
    "../results/global_bias.npy",
    np.array([global_bias])
)

print("\nFinal global model saved.")


# Final Global Model Evaluation

print("\nFINAL GLOBAL MODEL")

print("Using threshold: 0.5")

all_actual_values = []
all_predictions = []
all_probabilities = []


# Evaluate the same global model on each client

for client in clients:

    name = client["name"]

    X_test = client["X_test"]
    y_test = client["y_test"]

    linear_output = (
        np.dot(X_test, global_weights)
        + global_bias
    )

    probabilities = sigmoid(linear_output)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    all_actual_values.extend(y_test)
    all_predictions.extend(predictions)
    all_probabilities.extend(probabilities)

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

    print("\n" + name)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix:")
    print(matrix)


# Overall Final Global Model Evaluation

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