"""
Elevate Labs - Task 6
K-Nearest Neighbors Classification

Stage 2:
Dataset loading, feature selection, train-test split
and feature normalization.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Iris.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("IRIS DATASET")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nClass distribution:")
print(df["Species"].value_counts())


# Select meaningful numerical features
feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

X = df[feature_columns]
y = df["Species"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Normalize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFeature normalization completed.")

print("\nFirst normalized training sample:")
print(X_train_scaled[0])