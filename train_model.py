import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

print("1. Loading authentic DFT dataset...")
df = pd.read_csv("perovskite_bulk_energy.csv")

# Explicit column selection
formula_col = 'formula'
target_col = 'e_form'

print("2. Extracting composition descriptors (Elemental Fractions)...")


def get_fractions(formula):
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', str(formula))
    comp = {}
    for el, amt in matches:
        comp[el] = float(amt) if amt else 1.0
    total = sum(comp.values())
    return {el: amt/total for el, amt in comp.items()} if total > 0 else {}


X = pd.DataFrame(df[formula_col].apply(get_fractions).tolist()).fillna(0)
y = df[target_col]

print(
    f" -> Mapped {len(X.columns)} unique elements across {len(df)} perovskites.")

print("3. Preparing data and training models...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

print("\n4. Final Validation Results (20% Unseen Data):")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(
        f" -> {name}: R2 = {r2:.3f} | RMSE = {rmse:.3f} eV/atom | MAE = {mae:.3f} eV/atom")
