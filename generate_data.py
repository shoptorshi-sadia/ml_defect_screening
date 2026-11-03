import pandas as pd
import numpy as np

# Ensure reproducibility
np.random.seed(42)
n_defects = 500

# 1. Generate physical features of the defects
# Charge states typically range from -2 to +2
charge_state = np.random.choice([-2, -1, 0, 1, 2], n_defects)

# Ratio of the substituted ion's radius to the original ion (1.0 = perfect match)
radius_mismatch = np.random.uniform(0.85, 1.15, n_defects) 

# Difference in electronegativity (Pauling scale)
electro_diff = np.random.uniform(0.0, 1.2, n_defects)

# 2. Simulate the Formation Energy (eV)
# In physics, higher mismatch and higher electronegativity differences cost more energy.
# We add a small random noise to simulate DFT calculation variations.
base_energy = 1.2  # Base energy cost to form a defect
strain_penalty = 5.0 * np.abs(radius_mismatch - 1.0)
chemical_penalty = 0.8 * electro_diff
noise = np.random.normal(0, 0.1, n_defects)

formation_energy = base_energy + strain_penalty + chemical_penalty + noise

# 3. Package into a DataFrame and save
df = pd.DataFrame({
    'Charge_State': charge_state,
    'Radius_Mismatch': radius_mismatch,
    'Electronegativity_Diff': electro_diff,
    'Formation_Energy_eV': formation_energy
})

df.to_csv("defect_data.csv", index=False)
print("Successfully generated dataset: defect_data.csv with 500 simulated DFT calculations.")