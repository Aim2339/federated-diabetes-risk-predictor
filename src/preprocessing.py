import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_and_preprocess():
    # Load datasets
    bangladesh = pd.read_csv("../dataset/diabetes_bangladesh.csv")
    iraq = pd.read_csv("../dataset/diabetes_iraq.csv")
    pima = pd.read_csv("../dataset/diabetes_pima.csv")

    print("Original dataset sizes:")
    print("Bangladesh:", bangladesh.shape)
    print("Iraq:", iraq.shape)
    print("Pima:", pima.shape)

    # Bangladesh
    bangladesh = bangladesh[
        [
            "age",
            "bmi",
            "systolic_bp",
            "glucose",
            "diabetic"
        ]
    ].copy()

    bangladesh.columns = [
        "Age",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Diabetes"
    ]

    bangladesh["Diabetes"] = (
        bangladesh["Diabetes"]
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )

    bangladesh = bangladesh.dropna()

    # Iraq
    iraq["BloodPressure"] = (
        iraq["BP"]
        .astype(str)
        .str.split("/")
        .str[0]
    )

    iraq["BloodPressure"] = pd.to_numeric(
        iraq["BloodPressure"],
        errors="coerce"
    )

    iraq = iraq[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "RBS",
            "HbA1c"
        ]
    ].copy()

    iraq.columns = [
        "Age",
        "BMI",
        "BloodPressure",
        "Glucose",
        "HbA1c"
    ]

    iraq["Diabetes"] = (
        iraq["HbA1c"] >= 6.5
    ).astype(int)

    iraq = iraq.dropna()

    iraq = iraq[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "Glucose",
            "Diabetes"
        ]
    ]

    # Pima
    pima = pima[
        [
            "Age",
            "Body mass index",
            "Blood pressure",
            "Glucose",
            "Outcome"
        ]
    ].copy()

    pima.columns = [
        "Age",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Diabetes"
    ]

    pima[
        [
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ] = pima[
        [
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ].replace(0, pd.NA)

    pima = pima.dropna()


    # Display sizes after preprocessing
    print("\nAfter preprocessing:")
    print("Bangladesh:", bangladesh.shape)
    print("Iraq:", iraq.shape)
    print("Pima:", pima.shape)


    # Bangladesh train/test split
    X_bangladesh = bangladesh[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ]

    y_bangladesh = bangladesh["Diabetes"]
    (
        X_bangladesh_train,
        X_bangladesh_test,
        y_bangladesh_train,
        y_bangladesh_test
    ) = train_test_split(
        X_bangladesh,
        y_bangladesh,
        test_size=0.2,
        random_state=42,
        stratify=y_bangladesh
    )

    # Oversample Bangladesh class 1
    # Only training data is oversampled

    bangladesh_train = pd.concat([
        X_bangladesh_train.assign(
            Diabetes=y_bangladesh_train
        )
    ])

    class_0 = bangladesh_train[
        bangladesh_train["Diabetes"] == 0
    ]

    class_1 = bangladesh_train[
        bangladesh_train["Diabetes"] == 1
    ]

    class_1_oversampled = resample(
        class_1,
        replace=True,
        n_samples=len(class_0) // 3,
        random_state=42
    )

    bangladesh_train = pd.concat([
        class_0,
        class_1_oversampled
    ])

    bangladesh_train = bangladesh_train.sample(
        frac=1,
        random_state=42
    )

    X_bangladesh_train = bangladesh_train[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ]

    y_bangladesh_train = bangladesh_train["Diabetes"]

    # Scale Bangladesh data
    scaler_bangladesh = StandardScaler()
    X_bangladesh_train = scaler_bangladesh.fit_transform(
        X_bangladesh_train
    )
    X_bangladesh_test = scaler_bangladesh.transform(
        X_bangladesh_test
    )

    # Iraq train/test split
    X_iraq = iraq[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ]

    y_iraq = iraq["Diabetes"]

    (
        X_iraq_train,
        X_iraq_test,
        y_iraq_train,
        y_iraq_test
    ) = train_test_split(
        X_iraq,
        y_iraq,
        test_size=0.2,
        random_state=42,
        stratify=y_iraq
    )

    scaler_iraq = StandardScaler()
    X_iraq_train = scaler_iraq.fit_transform(
        X_iraq_train
    )
    X_iraq_test = scaler_iraq.transform(
        X_iraq_test
    )

    # Pima train/test split
    X_pima = pima[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "Glucose"
        ]
    ]

    y_pima = pima["Diabetes"]
    (
        X_pima_train,
        X_pima_test,
        y_pima_train,
        y_pima_test
    ) = train_test_split(
        X_pima,
        y_pima,
        test_size=0.2,
        random_state=42,
        stratify=y_pima
    )

    scaler_pima = StandardScaler()
    X_pima_train = scaler_pima.fit_transform(
        X_pima_train
    )
    X_pima_test = scaler_pima.transform(
        X_pima_test
    )

    # Print client information
    print("\nBangladesh:")
    print("Training:", X_bangladesh_train.shape)
    print("Testing:", X_bangladesh_test.shape)
    print("Class distribution:")
    print(
        y_bangladesh_train.value_counts()
    )
    print("\nIraq:")
    print("Training:", X_iraq_train.shape)
    print("Testing:", X_iraq_test.shape)
    print("Class distribution:")
    print(
        y_iraq_train.value_counts()
    )
    print("\nPima:")
    print("Training:", X_pima_train.shape)
    print("Testing:", X_pima_test.shape)
    print("Class distribution:")
    print(
        y_pima_train.value_counts()
    )

    # Create Bangladesh client
    bangladesh_client = {
        "name": "Bangladesh",
        "X_train": X_bangladesh_train,
        "y_train": y_bangladesh_train.to_numpy(),
        "X_test": X_bangladesh_test,
        "y_test": y_bangladesh_test.to_numpy()
    }

    # Create Iraq client
    iraq_client = {
        "name": "Iraq",
        "X_train": X_iraq_train,
        "y_train": y_iraq_train.to_numpy(),
        "X_test": X_iraq_test,
        "y_test": y_iraq_test.to_numpy()
    }

    # Create Pima client
    pima_client = {
        "name": "Pima",
        "X_train": X_pima_train,
        "y_train": y_pima_train.to_numpy(),
        "X_test": X_pima_test,
        "y_test": y_pima_test.to_numpy()
    }

    # Return all clients
    return [
        bangladesh_client,
        iraq_client,
        pima_client
    ]

if __name__ == "__main__":
    load_and_preprocess()