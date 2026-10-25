import kagglehub
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
path = kagglehub.dataset_download("tanuprabhu/linear-regression-dataset")
print("Dataset path:", path)

files = os.listdir(path)
print("Files:", files)

# 2. Read the CSV file
csv_file = [f for f in files if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

# 3. Inspect the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# 4. Separate input and target
X = df[["X"]]
Y = df["Y"]

# 5. Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)

# 6. Create the model
regression = LinearRegression()

# 7. Train the model
regression.fit(X_train, Y_train)

print("\nCoefficient:", regression.coef_)
print("Intercept:", regression.intercept_)

# 8. Make predictions
Y_pred = regression.predict(X_test)

print("\nActual values:")
print(Y_test.head())

print("\nPredicted values:")
print(Y_pred[:5])

# 9. Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

print("\nModel Evaluation:")
print("Test MAE:", mae)
print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test R2 Score:", r2)
