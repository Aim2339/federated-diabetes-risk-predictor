from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load all three clients
clients = load_and_preprocess()

# Train and evaluate each client separately
for client in clients:

    name = client["name"]

    X_train = client["X_train"]
    X_test = client["X_test"]

    y_train = client["y_train"]
    y_test = client["y_test"]

    print("\n" + name)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Get probabilities for class 1
    probabilities = model.predict_proba(X_test)[:, 1]

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
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("\nConfusion Matrix:")
    print(matrix)