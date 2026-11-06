import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

print("1. Loading authentic DFT dataset...")
df = pd.read_csv("defect_data.csv", nrows=5000)
formula_col = df.columns[0]
target_col = df.columns[1]

print("2. Extracting chemical fractions (Pure Python Mode)...")


def get_fractions(formula):
    # This reads strings like 'CsPbI3' and counts the atoms (Cs:1, Pb:1, I:3)
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', str(formula))
    comp = {}
    for el, amt in matches:
        comp[el] = float(amt) if amt else 1.0

    # Convert to percentages
    total = sum(comp.values())
    return {el: amt/total for el, amt in comp.items()} if total > 0 else {}


# Apply the custom parser
fractions_list = df[formula_col].apply(get_fractions).tolist()
X = pd.DataFrame(fractions_list).fillna(0)
y = df[target_col]

print(f" -> Successfully mapped {len(X.columns)} unique elements.")

print("3. Preparing data and training models...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

print("\n4. Final Results:")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    print(f" -> {name}: R2 = {r2:.3f} | RMSE = {rmse:.3f} eV")
