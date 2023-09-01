import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv('student/student-mat.csv', sep=';')

# Select features and target
features = ['age', 'Medu', 'Fedu', 'studytime'] # Add the attributes you want
target = 'G3'

# Split data into features (X) and target (y)
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict student performance on the testing set
y_pred = model.predict(X_test)

# Evaluate the model (you can use different evaluation metrics)
accuracy = model.score(X_test, y_test)

# Data Visualization

# Scatter plots for selected features vs. G3
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Relationship between Features and Student Performance (G3)')

axs[0, 0].scatter(X_test['age'], y_test, color='blue', label='Actual')
axs[0, 0].scatter(X_test['age'], y_pred, color='red', label='Predicted')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('G3')
axs[0, 0].legend()

axs[0, 1].scatter(X_test['Medu'], y_test, color='blue', label='Actual')
axs[0, 1].scatter(X_test['Medu'], y_pred, color='red', label='Predicted')
axs[0, 1].set_xlabel('Mother\'s Education')
axs[0, 1].set_ylabel('G3')
axs[0, 1].legend()

axs[1, 0].scatter(X_test['Fedu'], y_test, color='blue', label='Actual')
axs[1, 0].scatter(X_test['Fedu'], y_pred, color='red', label='Predicted')
axs[1, 0].set_xlabel('Father\'s Education')
axs[1, 0].set_ylabel('G3')
axs[1, 0].legend()

axs[1, 1].scatter(X_test['studytime'], y_test, color='blue', label='Actual')
axs[1, 1].scatter(X_test['studytime'], y_pred, color='red', label='Predicted')
axs[1, 1].set_xlabel('Weekly Study Time')
axs[1, 1].set_ylabel('G3')
axs[1, 1].legend()

# Actual vs. Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs. Predicted Student Performance (G3)')
plt.show()

print(f'Model Accuracy: {accuracy * 100:.2f}%')
# Extract coefficients from the trained model
coef_age = model.coef_[0]
coef_Medu = model.coef_[1]
coef_Fedu = model.coef_[2]
coef_studytime = model.coef_[3]

# Print the coefficients and their interpretations
print(f"Coefficient for Age: {coef_age:.2f}")
print(f"Coefficient for Mother's Education (Medu): {coef_Medu:.2f}")
print(f"Coefficient for Father's Education (Fedu): {coef_Fedu:.2f}")
print(f"Coefficient for Weekly Study Time (studytime): {coef_studytime:.2f}")
