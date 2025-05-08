# Predicting test scores from hours studied using linear regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("\nPredicting Test Scores Based on Study Time")

# Step 1: Create dataset
hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([20, 35, 50, 55, 65, 70, 80, 85, 90, 92, 99])
#hours = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 1000])
#scores = np.array([5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 95, 97, 99, 100])
#scores = scores + np.random.normal(0, 5, size=scores.shape)

# Step 2: Reshape and split into train/test datasets
X = hours.reshape(-1, 1)
y = scores

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = LinearRegression().fit(X_train, y_train)

# Step 4: Show learned equation
print(f"\nLearned equation: score = {model.coef_[0]:.2f} × hours + {model.intercept_:.2f}")

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("\nTest Predictions:")
for hrs, actual, pred in zip(X_test.flatten(), y_test, y_pred):
    print(f"  {hrs} hrs > Actual: {actual}% | Predicted: {pred:.1f}%")

print(f"\nModel R² score on test data: {model.score(X_test, y_test):.2f}")

# Step 6: Plot
plt.scatter(X, y, color='blue', label='Actual scores')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel("Hours Studied")
plt.ylabel("Test Score (%)")
plt.title("Study Time vs Test Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#plt.savefig("study_hard.png")
