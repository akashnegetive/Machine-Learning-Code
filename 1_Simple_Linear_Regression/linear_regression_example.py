import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate random dataset (200 rows)
np.random.seed(0)
hours = np.random.randint(1, 10, 200)
score = hours * 5 + np.random.randint(-10, 10, 200)

# Step 2: Create DataFrame
data = pd.DataFrame({"Hours": hours, "Score": score})

# Step 3: Separate input (X) and output (Y)
x = data[["Hours"]]
y = data["Score"]

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Step 5: Predict values for all Hours
y_pred = model.predict(x)

# Step 6: Add predictions to DataFrame
data["Predicted_Score"] = y_pred

# Step 7: Save dataset with predictions into CSV
data.to_csv("data200_with_predictions.csv", index=False)

# Step 8: Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 9: Plot actual vs predicted
plt.scatter(x, y, color="blue", alpha=0.5, label="Actual Scores")
plt.plot(x, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()
