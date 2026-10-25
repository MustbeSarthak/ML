# 📈 Simple Linear Regression Using Scikit-learn

## 📌 Overview

This project implements a Simple Linear Regression model using Python and Scikit-learn. It covers the complete machine learning workflow, including dataset loading, data inspection, train-test splitting, model training, prediction, and evaluation.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* KaggleHub
* Matplotlib (for visualization, if implemented)

## 📂 Dataset Information

* **Dataset:** [Linear Regression Dataset — Kaggle](https://www.kaggle.com/datasets/tanuprabhu/linear-regression-dataset)
* **Total Samples:** 300
* **Total Columns:** 2
* **Input Feature:** \(X\)
* **Target Variable:** \(Y\)
* **Missing Values:** 0
* **Duplicate Rows:** 0

## ⚙️ Implementation Workflow

1. Downloaded the dataset using KaggleHub.
2. Loaded the CSV file using Pandas.
3. Inspected the dataset using `head()`, `shape`, `info()`, and `describe()`.
4. Checked for missing values and duplicate rows.
5. Separated input features and the target variable.
6. Split the dataset into training and testing sets.
7. Trained the Linear Regression model.
8. Generated predictions on the test dataset.
9. Evaluated the model using MAE, MSE, RMSE, and R² Score.

## 🧮 Linear Regression Equation

Simple Linear Regression estimates the relationship between an input variable and a target variable using:

$$
\hat{Y} = \beta_0 + \beta_1 X
$$

Where:

* \(\hat{Y}\) = Predicted target value
* \(X\) = Input feature
* \(\beta_0\) = Intercept
* \(\beta_1\) = Regression coefficient (slope)

### Learned Model Parameters

* **Coefficient:** \(0.65023172\)
* **Intercept:** \(4.86580154\)

The trained model equation is:

$$
\boxed{\hat{Y} = 4.86580154 + 0.65023172X}
$$

## 📊 Model Configuration

| Parameter        |             Value |
| ---------------- | ----------------: |
| Algorithm        | Linear Regression |
| Total Samples    |               300 |
| Training Samples |               240 |
| Testing Samples  |                60 |
| Training Ratio   |               80% |
| Testing Ratio    |               20% |
| Random State     |                 0 |

## 📉 Model Evaluation Results

The model was evaluated on the test dataset using the following metrics.

| Evaluation Metric              |    Score |
| ------------------------------ | -------: |
| Mean Absolute Error (MAE)      |   4.6049 |
| Mean Squared Error (MSE)       | 656.1609 |
| Root Mean Squared Error (RMSE) |  25.6156 |
| R² Score                       |   0.7986 |

### 1. Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**Result:**

$$
\mathrm{MAE} \approx 4.6049
$$

### 2. Mean Squared Error (MSE)

MSE calculates the average of the squared prediction errors.

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Result:**

$$
\mathrm{MSE} \approx 656.1609
$$

### 3. Root Mean Squared Error (RMSE)

RMSE is the square root of the Mean Squared Error.

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

**Result:**

$$
\mathrm{RMSE} \approx 25.6156
$$

### 4. R² Score (Coefficient of Determination)

R² measures how much of the variation in the target variable is explained by the model.

$$
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

Where:

* \(y_i\) = Actual value
* \(\hat{y}_i\) = Predicted value
* \(\bar{y}\) = Mean of actual values
* \(n\) = Number of test samples

**Result:**

$$
R^2 \approx 0.7986
$$

The model explains approximately **79.86% of the variation** in the target variable on the test dataset. This does not mean the model has 79.86% prediction accuracy.

## 🎯 Key Learnings

* Understanding the fundamentals of Simple Linear Regression.
* Loading and inspecting datasets using Pandas.
* Separating input features and target variables.
* Splitting data into training and testing sets.
* Training models using Scikit-learn.
* Generating predictions on unseen test data.
* Evaluating regression models using MAE, MSE, RMSE, and R².
* Interpreting regression coefficients and model performance.

## 🚀 Future Improvements

* Visualize actual versus predicted values.
* Plot the regression line against the dataset.
* Perform residual analysis to investigate large prediction errors.
* Compare performance across different train-test splits.

---

**Project Status:** Completed — Linear Regression implementation, prediction, and evaluation.
