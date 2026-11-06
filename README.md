# Machine Learning for DFT-Derived Bulk Formation Energy Prediction in Perovskites

## 📌 Scientific Objective
This project implements a machine learning pipeline to predict the **bulk formation energy** ($E_{form}$) of halide and oxide perovskites. By mapping chemical compositions to thermodynamic stability, this repository serves as a baseline materials screening tool.

*(Note: This project previously targeted point-defect screening but has been pivoted to bulk perovskite formation energies to align with the selected dataset).*

## 📂 Dataset Provenance
- **Source:** `matbench_perovskites` (via Matminer)
- **Size:** 18,928 authentic Density Functional Theory (DFT) calculations.
- **Target Variable:** `e_form` (Formation energy in eV/cell).

## ⚙️ Methodology & Limitations
Due to local hardware memory constraints (`MemoryError` during complex structural featurization), this pipeline utilizes **composition-based descriptors** rather than full structural representations. 
- 3D `pymatgen` structures were reduced to chemical formulas.
- Formulas were safely parsed using `pymatgen.core.Composition` to extract exact elemental fractions.
- Models were trained to map elemental stoichiometry directly to quantum mechanical formation energies.

## 📊 Validation & Performance
The models were evaluated using a 20% held-out test set with explicit column mapping (`formula`, `e_form`). To contextualize predictive power, 5-fold cross-validation and a baseline `DummyRegressor` (mean prediction) were implemented in the pipeline.

- **Best Performing Model:** XGBoost
- **Test R² Score:** 0.313
- **Test RMSE:** 0.621 eV/cell
- **Test MAE:** 0.462 eV/cell

## 🔬 Feature Importance & Interpretability
![Feature Importance](feature_importance.png)

The model identifies certain elemental composition features (such as the presence of Rb, Cs, and K) as influential predictors of the formation energy within this dataset. 

*Note: These associations should be interpreted cautiously; they establish predictive correlations based purely on composition rather than directly quantifying crystallographic stability metrics like the Goldschmidt tolerance factor.*

## 🚀 Reproducibility
1. Install dependencies: `pip install -r requirements.txt`
2. Fetch the data: `python fetch_dft_data.py` (Generates `perovskite_bulk_energy.csv`)
3. Train models: `python train_model.py`
4. Generate validation plots: `python plot_validation.py`