import pandas as pd
import re
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

print("Generating Scientific Visualization...")
df = pd.read_csv("perovskite_bulk_energy.csv")


def get_fractions(formula):
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', str(formula))
    comp = {}
    for el, amt in matches:
        comp[el] = float(amt) if amt else 1.0
    total = sum(comp.values())
    return {el: amt/total for el, amt in comp.items()} if total > 0 else {}


X = pd.DataFrame(df['formula'].apply(get_fractions).tolist()).fillna(0)
y = df['e_form']

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
top_10 = importance.sort_values(ascending=True).tail(10)

plt.figure(figsize=(8, 5))
top_10.plot(kind='barh', color='#2b83ba')
plt.title("XGBoost Feature Importance: Drivers of Bulk Formation Energy")
plt.xlabel("Relative Importance")
plt.ylabel("Elemental Fraction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("-> Success! 'feature_importance.png' has been updated.")
