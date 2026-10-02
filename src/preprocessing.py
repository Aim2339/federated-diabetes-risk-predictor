import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def prepare_client(data, oversample=False):
    X = data[
        ["Age", "BMI", "BloodPressure", "Glucose"]
    ]
    y = data["Diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Oversample only the training data
    if oversample:
        train_data = X_train.copy()
        train_data["Diabetes"] = y_train.values

        class_0 = train_data[
            train_data["Diabetes"] == 0
        ]

        class_1 = train_data[
            train_data["Diabetes"] == 1
        ]

        class_1 = resample(
            class_1,
            replace=True,
            n_samples=len(class_0) // 3,
            random_state=42
        )

        train_data = pd.concat([
            class_0,
            class_1
        ])

        train_data = train_data.sample(
            frac=1,
            random_state=42
        )

        X_train = train_data[
            ["Age", "BMI", "BloodPressure", "Glucose"]
        ]

        y_train = train_data["Diabetes"]

    # Scale the data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train.to_numpy(),
        y_test.to_numpy()
    )

def load_and_preprocess():
    # Load datasets
    bangladesh = pd.read_csv(
        "../dataset/diabetes_bangladesh.csv"
    )

    iraq = pd.read_csv(
        "../dataset/diabetes_iraq.csv"
    )

    pima = pd.read_csv(
        "../dataset/diabetes_pima.csv"
    )

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

    iraq["Diabetes"] = (
        iraq["HbA1c"] >= 6.5
    ).astype(int)

    iraq = iraq[
        [
            "Age",
            "BMI",
            "BloodPressure",
            "RBS",
            "Diabetes"
        ]
    ].copy()

    iraq = iraq.rename(
        columns={"RBS": "Glucose"}
    )

    iraq = iraq.dropna()

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
        ["BMI", "BloodPressure", "Glucose"]
    ] = pima[
        ["BMI", "BloodPressure", "Glucose"]
    ].replace(0, pd.NA)

    pima = pima.dropna()

    # Prepare clients
    bangladesh_data = prepare_client(
        bangladesh,
        oversample=True
    )

    iraq_data = prepare_client(
        iraq
    )

    pima_data = prepare_client(
        pima
    )

    # Display client information
    print("\nAfter preprocessing:")
    print("\nBangladesh:")
    print("Training:", bangladesh_data[0].shape)
    print("Testing:", bangladesh_data[1].shape)
    print(
        "Class distribution:",
        pd.Series(bangladesh_data[2]).value_counts()
    )

    print("\nIraq:")
    print("Training:", iraq_data[0].shape)
    print("Testing:", iraq_data[1].shape)
    print(
        "Class distribution:",
        pd.Series(iraq_data[2]).value_counts()
    )

    print("\nPima:")
    print("Training:", pima_data[0].shape)
    print("Testing:", pima_data[1].shape)
    print(
        "Class distribution:",
        pd.Series(pima_data[2]).value_counts()
    )

    # Create clients
    clients = [

        {
            "name": "Bangladesh",
            "X_train": bangladesh_data[0],
            "y_train": bangladesh_data[2],
            "X_test": bangladesh_data[1],
            "y_test": bangladesh_data[3]
        },

        {
            "name": "Iraq",
            "X_train": iraq_data[0],
            "y_train": iraq_data[2],
            "X_test": iraq_data[1],
            "y_test": iraq_data[3]
        },

        {
            "name": "Pima",
            "X_train": pima_data[0],
            "y_train": pima_data[2],
            "X_test": pima_data[1],
            "y_test": pima_data[3]
        }
    ]
    return clients

if __name__ == "__main__":
    load_and_preprocess()