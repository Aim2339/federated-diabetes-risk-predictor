import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load data
    df = pd.read_csv("../dataset/diabetes.csv")
    print("Original data:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)

    # Convert target to binary
    df["Diabetes_012"] = df["Diabetes_012"].replace({0: 0, 1: 1, 2: 1})

    print("\nTarget values:")
    print(df["Diabetes_012"].value_counts())

    # Separate features and target
    X = df.drop("Diabetes_012", axis=1)
    y = df["Diabetes_012"]

    print("\nNumber of features:", X.shape[1])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # Scale selected features
    features_to_scale = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

    scaler = StandardScaler()

    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

    print("\nFeatures scaled:")
    print(features_to_scale)

    print("\nPreprocessing complete!")

    return X_train, X_test, y_train, y_test