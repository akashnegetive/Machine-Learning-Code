import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Step 1: Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, 100)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Try models with higher polynomial degrees
degrees = range(1, 41)
train_errors = []
test_errors = []

for d in degrees:
    # Use pipeline: polynomial features + scaling + regression
    model = make_pipeline(PolynomialFeatures(degree=d), StandardScaler(with_mean=False), LinearRegression())
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, y_pred_train))
    test_errors.append(mean_squared_error(y_test, y_pred_test))

# Step 4: Find best degree (lowest test error)
best_degree = degrees[np.argmin(test_errors)]
best_error = min(test_errors)

# Step 5: Plot training vs test error
plt.figure(figsize=(12, 6))
plt.plot(degrees, train_errors, marker='o', label="Training Error", color="blue")
plt.plot(degrees, test_errors, marker='o', label="Test Error", color="red")

# Highlight the best degree
plt.axvline(best_degree, color="green", linestyle="--", label=f"Best Degree = {best_degree}")
plt.scatter(best_degree, best_error, color="black", zorder=5)
plt.text(best_degree+0.5, best_error+0.02, f"Best Degree = {best_degree}", color="green")

plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff with Optimal Model Highlighted")
plt.legend()
plt.grid(True)
plt.show()
