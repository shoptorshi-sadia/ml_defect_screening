import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.DataFrame(pd.read_csv("defect_data.csv"))

# 2. Separate Features (X) from the Target Answer (y)
X = df[['Charge_State', 'Radius_Mismatch', 'Electronegativity_Diff']]
y = df['Formation_Energy_eV']

# 3. Split data: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predict on the unseen test data
predictions = rf_model.predict(X_test)

# 6. Evaluate accuracy
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)  # 1.0 is a perfect score

print(f"Model Mean Squared Error: {mse:.4f} eV^2")
print(f"Model R-squared (Accuracy): {r2:.4f}")

# 7. Visualize the results: True vs. Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, predictions, alpha=0.6, color='darkorange')
plt.plot([y.min(), y.max()], [y.min(), y.max()],
         'k--', lw=2)  # The perfect prediction line
plt.xlabel('True Formation Energy (eV)', fontsize=12)
plt.ylabel('ML Predicted Energy (eV)', fontsize=12)
plt.title('Random Forest: DFT vs ML Predictions', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
