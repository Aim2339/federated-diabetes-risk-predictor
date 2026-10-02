# Federated Diabetes Risk Predictor

A simple **Federated Learning** project that predicts diabetes risk using data from three different datasets:

1. Bangladesh
2. Iraq
3. Pima

The main idea is to train a model for each client separately and then combine the model results using **Federated Averaging (FedAvg)**.

> **Note:** This is a simulated Federated Learning project. The three datasets are treated as separate clients/cohorts. They are not real hospitals connected to a federated server.

---

## 1. What is the Project About?

Normally, if we want to train one machine-learning model using several datasets, we could combine all the data into one place.

In Federated Learning:

1. Each client keeps its own data.
2. Each client trains the model using its own data.
3. The clients send model values instead of sending the original data.
4. These model values are combined.
5. A new global model is created.
6. The process is repeated for several rounds.

In this project, the three clients are:

- **Bangladesh**
- **Iraq**
- **Pima**

---

## 2. Main Objectives

1. Use diabetes datasets from different sources.
2. Find features that are available in all three datasets.
3. Clean and prepare the data.
4. Train a Logistic Regression model.
5. Train the model separately on each client.
6. Combine the client models using FedAvg.
7. Repeat the process for 10 rounds.
8. Check the final model using different evaluation measures.

---

## 3. Datasets

| Client | Dataset | Original Rows | Training Rows | Testing Rows |
|---|---|---:|---:|---:|
| Bangladesh | `diabetes_bangladesh.csv` | 5,288 | 5,274 | 1,058 |
| Iraq | `diabetes_iraq.csv` | 662 | 460 | 115 |
| Pima | `diabetes_pima.csv` | 768 | 579 | 145 |

### Important

The Bangladesh training data was balanced by increasing the number of diabetes samples.

The test data was **not** oversampled.

---

## 4. Common Features

The three datasets use different column names, so we selected four features that can be used by all three clients.

1. **Age**
2. **BMI**
3. **Blood Pressure**
4. **Glucose**

The output is:

- `0` → No Diabetes
- `1` → Diabetes

---

## 5. Data Preprocessing

The following steps were used:

1. Load the three CSV files.
2. Select the required columns.
3. Rename the columns so all three datasets use the same names.
4. Convert the diabetes output into `0` and `1`.
5. Remove missing values.
6. Convert the Iraq blood-pressure value such as `120/80` into the systolic value `120`.
7. For the Iraq dataset, diabetes was decided using:
   - `HbA1c >= 6.5` → Diabetes
   - `HbA1c < 6.5` → No Diabetes
8. In the Pima dataset, zero values for BMI, Blood Pressure, and Glucose were treated as missing values.
9. Split each dataset into:
   - 80% training
   - 20% testing
10. Scale the four input features using `StandardScaler`.
11. Only the Bangladesh training data was oversampled to improve its class balance.

---

## 6. Why Do We Use StandardScaler?

The four features have different value ranges.

For example:

- Age may be around 20–80.
- BMI may be around 15–50.
- Glucose may be around 50–300.

`StandardScaler` puts these values on a similar scale.

This helps the Logistic Regression model train properly.

---

## 7. Model Used

The final model is **Logistic Regression**.

It uses four input values:

```text
Age
BMI
BloodPressure
Glucose
```

The model calculates a probability between 0 and 1.

For example:

```text
0.20 → lower predicted risk
0.80 → higher predicted risk
```

A threshold of **0.5** is used:

- Probability >= 0.5 → Diabetes
- Probability < 0.5 → No Diabetes

---

## 8. How Federated Learning Works

The project uses **Federated Averaging (FedAvg)**.

### Each round works like this:

1. Start with the current global model.
2. Send the global model values to each client.
3. Bangladesh trains the model on its local data.
4. Iraq trains the model on its local data.
5. Pima trains the model on its local data.
6. Collect the updated model values.
7. Combine the client values using FedAvg.
8. Create the new global model.
9. Start the next round.

This is repeated for:

**10 rounds**

Each client performs:

**20 local training epochs**

Learning rate:

**0.01**

---

## 9. Federated Averaging

The global model is created by taking a weighted average of the client models.

In simple terms:

```text
Global Model =
Client 1 Model × Client 1 Weight
+
Client 2 Model × Client 2 Weight
+
Client 3 Model × Client 3 Weight
```

The weight depends on the number of training samples used by each client.

The current training sizes are:

- Bangladesh: 5,274
- Iraq: 460
- Pima: 579

Total:

**6,313 training samples**

---

## 10. Project Structure

```text
federated-diabetes-risk-predictor/
│
├── dataset/
│   ├── diabetes_bangladesh.csv
│   ├── diabetes_iraq.csv
│   └── diabetes_pima.csv
│
├── results/
│   ├── class_distribution.png
│   ├── client_metrics.png
│   ├── overall_metrics.png
│   └── confusion_matrix.png
│
├── src/
│   ├── preprocessing.py
│   ├── train_clients.py
│   ├── federated_train.py
│   └── graphs.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 11. Files in the Project

### `preprocessing.py`

This file:

1. Loads the three datasets.
2. Cleans the data.
3. Selects the four common features.
4. Splits the data into training and testing sets.
5. Balances the Bangladesh training data.
6. Scales the data.
7. Creates the three clients.

### `train_clients.py`

This file trains a normal Logistic Regression model separately for each client.

It is used as a **local baseline** for comparison.

### `federated_train.py`

This is the main Federated Learning file.

It:

1. Loads the clients.
2. Creates the initial global model.
3. Trains each client locally.
4. Combines the client models.
5. Repeats the process for 10 rounds.
6. Tests the final global model.

### `graphs.py`

This file creates the project graphs.

---

## 12. How to Run the Project

### Step 1: Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Install the required packages

```powershell
pip install -r requirements.txt
```

### Step 3: Run preprocessing

Go to the `src` folder:

```powershell
cd src
```

Then:

```powershell
python preprocessing.py
```

### Step 4: Run the local models

```powershell
python train_clients.py
```

### Step 5: Run Federated Learning

```powershell
python federated_train.py
```

### Step 6: Create the graphs

```powershell
python graphs.py
```

The graphs will be saved in the `results` folder.

---

## 13. Local Model Results

The clients were also tested separately using normal Logistic Regression.

| Client | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Bangladesh | 76.94% | 16.41% | 63.24% | 26.06% | 0.7754 |
| Iraq | 94.78% | 80.00% | 88.89% | 84.21% | 0.9599 |
| Pima | 68.97% | 53.73% | 72.00% | 61.54% | 0.7823 |

These results show how each client performs when it trains its own model without Federated Learning.

---

## 14. Final Federated Model Results

After 10 Federated Learning rounds:

| Client | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Bangladesh | 89.79% | 26.19% | 32.35% | 28.95% | 0.7723 |
| Iraq | 91.30% | 78.57% | 61.11% | 68.75% | 0.9439 |
| Pima | 73.10% | 65.71% | 46.00% | 54.12% | 0.7541 |

---

## 15. Overall Results

The final global model was tested on all three test sets together.

| Metric | Result |
|---|---:|
| Accuracy | **88.09%** |
| Precision | **42.11%** |
| Recall | **41.18%** |
| F1-Score | **41.64%** |
| ROC-AUC | **80.29%** |

Total test samples:

**1,318**

---

## 16. Confusion Matrix

The final confusion matrix is:

```text
                 Predicted
              No Diabetes  Diabetes

Actual
No Diabetes       1105        77
Diabetes            80        56
```

Therefore:

1. True Negatives = **1,105**
2. False Positives = **77**
3. False Negatives = **80**
4. True Positives = **56**

---

## 17. Experiments During the Project

Several approaches were tested before selecting the final setup.

### 1. MLP

A Neural Network was tested first.

The results were not good enough for the project, so other machine-learning models were tested.

### 2. Random Forest

Random Forest was tested as another traditional machine-learning model.

### 3. Logistic Regression

Logistic Regression was selected for the Federated Learning implementation because its model values (weights and bias) can be combined using FedAvg.

### 4. Gradient Boosting

Gradient Boosting was also tested as a local model.

However, its tree structures cannot simply be averaged using the same FedAvg method used for Logistic Regression.

### 5. Bangladesh Oversampling

Bangladesh had a much smaller number of diabetes samples.

The training data was therefore oversampled.

### 6. Learning Rate

Different learning rates were tested.

The final learning rate was:

**0.01**

### 7. Decision Threshold

Different prediction thresholds were tested.

The final threshold was:

**0.5**

### 8. Pima Oversampling

Pima oversampling was also tested.

It did not improve the final overall results, so the original Pima training data was kept.

---

## 18. Graphs

The project contains four graphs:

1. **Training Class Distribution**
2. **Client Performance**
3. **Overall Model Performance**
4. **Overall Confusion Matrix**

These graphs are stored in the `results` folder.

---

## 19. Important Limitations

There are some limitations in this project:

1. This is a **simulation** of Federated Learning.
2. The three clients are datasets/cohorts, not real hospitals.
3. The three datasets come from different sources and have different characteristics.
4. Only four common features were used.
5. The Iraq diabetes label was created using `HbA1c >= 6.5`.
6. The Pima dataset represents a specific population, so the results should not be treated as results for every population.
7. The final model has good overall accuracy, but its recall is only **41.18%**.
8. Therefore, the model still misses some actual diabetes cases.

---

## 20. Future Improvements

Possible future improvements include:

1. Use more common features between the datasets.
2. Test more machine-learning models.
3. Use larger datasets.
4. Improve the balance between the client datasets.
5. Use real separate machines or servers for the clients.
6. Add a user interface for predictions.
7. Test the model on new datasets.
8. Explore more advanced Federated Learning methods.

---

## 21. Technologies Used

- **Python**
- **Pandas** – used for handling the datasets.
- **NumPy** – used for numerical calculations.
- **Scikit-learn** – used for preprocessing, Logistic Regression, and evaluation.
- **Matplotlib** – used to create graphs.

---

## 22. Conclusion

This project demonstrates how Federated Learning can be used to train a diabetes-risk model using multiple separate datasets.

The main process was:

```text
Datasets
   ↓
Data Cleaning
   ↓
Common Features
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Local Training
   ↓
Federated Averaging
   ↓
Global Model
   ↓
Evaluation
```

The final Federated Learning model achieved:

**88.09% Accuracy**

and

**80.29% ROC-AUC**

on the combined test data.

The project also shows that using different datasets as separate clients can be used to demonstrate the basic idea of Federated Learning without combining the original datasets into one training dataset.
