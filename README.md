# Predicting-House-Prices-with-Linear-Regression-
 # Description:  The objective of this project is to build a predictive model using linear regression to estimate a numerical outcome based on a dataset with relevant features. Linear regression is a fundamental machine learning algorithm, and this project provides hands-on experience in developing, evaluating, and interpreting a predictive model
 #code
 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("housing dataset.csv")

print("\nFirst 5 rows of dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

data = data.dropna()

print("\nAfter removing missing values:")
print(data.shape)

binary_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_columns:
    data[col] = data[col].map({'yes':1, 'no':0})

data['furnishingstatus'] = data['furnishingstatus'].map({
    'furnished':2,
    'semi-furnished':1,
    'unfurnished':0
})

print("\nDataset after converting categorical values:")
print(data.head())

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


X = data.drop("price", axis=1)  
y = data["price"]                

print("\nFeatures used for training:")
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed")


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

feature = "area"

plt.figure(figsize=(8,6))
plt.scatter(data[feature], y)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price Relationship")
plt.show()

sample = X_test.iloc[0].values.reshape(1,-1)

predicted_price = model.predict(sample)

print("\nExample House Price Prediction:")
print("Predicted Price:", predicted_price[0])

