# Machine-Learning Accelerated Defect Screening

## Overview
This project bypasses the computational bottleneck of high-throughput Density Functional Theory (DFT) calculations by predicting point defect formation energies in perovskites using a Random Forest machine learning model.

## Workflow
1. **Data Generation:** Simulated a physical dataset of 500 defects based on charge state, ionic radius mismatch, and electronegativity differences.
2. **Model Training:** Trained a Random Forest Regressor using `scikit-learn` to map structural features to formation energy.
3. **Validation:** Evaluated using Mean Squared Error (MSE) and $R^2$ metrics against unseen test data.

## Installation