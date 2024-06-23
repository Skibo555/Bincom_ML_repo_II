import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('BostonHousing.csv')

# Check for missing values
df = df.dropna()

# Replacing the target variable for house prices
if 'medv' in df.columns:
    df.rename(columns={'medv': 'price'}, inplace=True)
else:
    raise ValueError("The dataset does not contain the target variable 'MEDV'.")

# Define features and target variable
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax']
X = df[features]
y = df['price']

# Handle missing values by imputing with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mean_ab_err = mean_absolute_error(y_test, y_pred)
mean_sq_err = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_sq_err)
r2 = r2_score(y_test, y_pred)

# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mean_ab_err}")
print(f"Mean Squared Error (MSE): {mean_sq_err}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
# print(f"This is the level of accuracy of the model: {accuracy}")
# print(f"Confusion Matrix {conf_matrix}")
# print(f"Classification report: {class_report}")

