import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from pymatgen.core import Composition
import warnings

warnings.filterwarnings("ignore")

print("Generating Validation Plots...")
df = pd.read_csv("perovskite_bulk_energy.csv")


def get_fractions(formula):
    try:
        return Composition(formula).fractional_composition.get_el_amt_dict()
    except:
        return {}


X = pd.DataFrame(df['formula'].apply(get_fractions).tolist()).fillna(0)
y = df['e_form']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Parity Plot
ax1.scatter(y_test, y_pred, alpha=0.4, color='#2b83ba')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel("DFT Formation Energy (eV/cell)")
ax1.set_ylabel("Predicted Formation Energy (eV/cell)")
ax1.set_title("Parity Plot: XGBoost vs DFT")

# Residual Plot
residuals = y_test - y_pred
ax2.hist(residuals, bins=30, color='#abdda4', edgecolor='black')
ax2.set_xlabel("Residual Error (eV/cell)")
ax2.set_ylabel("Frequency")
ax2.set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("model_validation.png", dpi=300)
print("-> Saved model_validation.png")
