from matminer.datasets import load_dataset
import pandas as pd

print("Downloading authentic perovskite DFT dataset...")

try:
    # This downloads a peer-reviewed dataset of 18,928 perovskite calculations
    df = load_dataset("matbench_perovskites")

    # Save it to a CSV file for our ML model
    df.to_csv("defect_data.csv", index=False)
    print(f"Success! Saved {len(df)} real DFT calculations to defect_data.csv")

except Exception as e:
    print(f"Error: {e}")
