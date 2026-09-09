# """
# Elevate Labs - Task 6
# K-Nearest Neighbors Classification

# Stage 2:
# Dataset loading, feature selection, train-test split
# and feature normalization.
# """

# from pathlib import Path

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_PATH = BASE_DIR / "data" / "Iris.csv"


# # Load dataset
# df = pd.read_csv(DATA_PATH)

# print("=" * 60)
# print("IRIS DATASET")
# print("=" * 60)

# print("\nDataset shape:")
# print(df.shape)

# print("\nFirst five rows:")
# print(df.head())

# print("\nClass distribution:")
# print(df["Species"].value_counts())


# # Select meaningful numerical features
# feature_columns = [
#     "SepalLengthCm",
#     "SepalWidthCm",
#     "PetalLengthCm",
#     "PetalWidthCm",
# ]

# X = df[feature_columns]
# y = df["Species"]


# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )


# # Normalize features
# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# print("\nTraining samples:", len(X_train))
# print("Testing samples:", len(X_test))

# print("\nFeature normalization completed.")

# print("\nFirst normalized training sample:")
# print(X_train_scaled[0])

"""
Elevate Labs - Task 6
K-Nearest Neighbors Classification

Stage 3:
Train KNN models with different K values
and determine the best K using accuracy.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Iris.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

X = df[features]
y = df["Species"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Normalize
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Test different K values
k_values = range(1, 21)
accuracies = []

print("=" * 60)
print("KNN K-VALUE EXPERIMENT")
print("=" * 60)

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracies.append(accuracy)

    print(
        f"K = {k:2d} | "
        f"Accuracy = {accuracy:.4f}"
    )


# Select best K
best_index = accuracies.index(
    max(accuracies)
)

best_k = list(k_values)[best_index]

print("\n" + "=" * 60)
print("BEST K")
print("=" * 60)

print("Best K:", best_k)
print(
    "Best Accuracy:",
    f"{accuracies[best_index]:.4f}"
)