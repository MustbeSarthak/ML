from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Initializing the Ridge
ridge = Ridge(alpha = 1)

#Loading the data
data = load_diabetes()

# Processing the data
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Implementing GridSearchCV
parameter = {'alpha': np.logspace(-3, 3, 100)}


# Hyper Parameter Tuning
ridgeCV = GridSearchCV(ridge, parameter, scoring = 'neg_mean_squared_error',cv = 5)

ridgeCV.fit(X_train, Y_train)

print(ridgeCV.best_params_)
print(ridgeCV.best_score_)


#Prediction
Y_pred = ridgeCV.predict(X_test)
print("Best Alpha:", ridgeCV.best_params_)
print("Test R2:", r2_score(Y_test, Y_pred))
print("Test MAE:", mean_absolute_error(Y_test, Y_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))


# Graph
plt.figure(figsize=(8, 6))

plt.scatter(Y_test, Y_pred, alpha=0.6, label="Predictions")

# Perfect prediction line
plt.plot([Y_test.min(), Y_test.max()],[Y_test.min(), Y_test.max()],color="red",linestyle="--",label="Perfect Prediction")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)

plt.show()