import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

RAW_PATH = Path("data/raw/insurance.csv")
OUT_PATH = Path("data/synthetic/insurance_large.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

base = pd.read_csv(RAW_PATH)

# Expand dataset (BIG DATA simulation)
multiplier = 500
large = pd.concat([base] * multiplier, ignore_index=True)

# Add patient ID
large["patient_id"] = np.arange(1, len(large) + 1)

# Add noise
large["age"] = np.clip(
    large["age"] + np.random.randint(-2, 3, size=len(large)),
    18, 64
)

large["bmi"] = np.clip(
    large["bmi"] + np.random.normal(0, 1.5, size=len(large)),
    15, 55
)

# Simulate healthcare behavior
large["primary_care_visits"] = np.clip(
    (large["age"] / 20 + large["children"] + np.random.poisson(1, len(large))).astype(int),
    0, 12
)

large["emergency_visits"] = np.clip(
    (large["smoker"].map({"yes": 2, "no": 0}) + np.random.poisson(0.5, len(large))).astype(int),
    0, 8
)

large["hospital_visits"] = np.clip(
    (large["primary_care_visits"] * 0.3 + large["emergency_visits"] + np.random.poisson(1, len(large))).astype(int),
    0, 15
)

# Preventive behavior
large["preventive_visit_flag"] = (large["primary_care_visits"] >= 2).astype(int)

# Health score
large["chronic_condition_score"] = np.clip(
    ((large["bmi"] / 10) + (large["age"] / 20) + large["smoker"].map({"yes": 2, "no": 0})).astype(int),
    0, 10
)

large["medication_count"] = np.clip(
    (large["chronic_condition_score"] + np.random.poisson(1, len(large))).astype(int),
    0, 15
)

# High-cost label
threshold = large["charges"].quantile(0.80)
large["high_cost"] = (large["charges"] > threshold).astype(int)

# Preventable cases
large["preventable_case"] = (
    (large["high_cost"] == 1) &
    (large["primary_care_visits"] <= 1) &
    (large["preventive_visit_flag"] == 0)
).astype(int)

# Save dataset
large.to_csv(OUT_PATH, index=False)

print("✅ Dataset created!")
print("Shape:", large.shape)
print(large.head())