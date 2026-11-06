from matminer.datasets import load_dataset
import pandas as pd

print("1. Fetching authentic Matbench perovskites dataset...")
df = load_dataset("matbench_perovskites")

print("2. Extracting chemical formulas from 3D structures...")
# Safely extract the formula from the pymatgen structure object
df['formula'] = df['structure'].apply(lambda s: s.composition.reduced_formula)

# Keep only explicit, named columns
df_clean = df[['formula', 'e_form']]

# Save to a newly named file to reflect the new project target
df_clean.to_csv("perovskite_bulk_energy.csv", index=False)
print("Success! Clean dataset saved to perovskite_bulk_energy.csv")
