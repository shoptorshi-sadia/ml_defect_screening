import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from pymatgen.core import Composition
import warnings

warnings.filterwarnings("ignore")

print("1. Loading authentic DFT dataset...")
df = pd.read_csv("perovskite_bulk_energy.csv")

print("2. Extracting composition descriptors using pymatgen...")


def get_fractions(formula):
    try:
        return Composition(formula).fractional_composition.get_el_amt_dict()
    except Exception:
        return {}


X = pd.DataFrame(df['formula'].apply(get_fractions).tolist()).fillna(0)
y = df['e_form']
print(f" -> Mapped {len(X.columns)} unique elements.")

print("3. Preparing data and K-Fold Cross-Validation...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Baseline (Mean)": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

print("\n4. Final Validation Results:")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print(f" -> {name}:")
    print(
        f"    CV R2 (5-fold): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(
        f"    Test R2: {r2:.3f} | Test RMSE: {rmse:.3f} eV/cell | Test MAE: {mae:.3f} eV/cell")
