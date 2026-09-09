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

# """
# Elevate Labs - Task 6
# K-Nearest Neighbors Classification

# Stage 3:
# Train KNN models with different K values
# and determine the best K using accuracy.
# """

# from pathlib import Path

# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score


# BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_PATH = BASE_DIR / "data" / "Iris.csv"


# # Load dataset
# df = pd.read_csv(DATA_PATH)

# # Features and target
# features = [
#     "SepalLengthCm",
#     "SepalWidthCm",
#     "PetalLengthCm",
#     "PetalWidthCm",
# ]

# X = df[features]
# y = df["Species"]


# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )


# # Normalize
# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# # Test different K values
# k_values = range(1, 21)
# accuracies = []

# print("=" * 60)
# print("KNN K-VALUE EXPERIMENT")
# print("=" * 60)

# for k in k_values:

#     model = KNeighborsClassifier(
#         n_neighbors=k
#     )

#     model.fit(
#         X_train_scaled,
#         y_train
#     )

#     predictions = model.predict(
#         X_test_scaled
#     )

#     accuracy = accuracy_score(
#         y_test,
#         predictions
#     )

#     accuracies.append(accuracy)

#     print(
#         f"K = {k:2d} | "
#         f"Accuracy = {accuracy:.4f}"
#     )


# # Select best K
# best_index = accuracies.index(
#     max(accuracies)
# )

# best_k = list(k_values)[best_index]

# print("\n" + "=" * 60)
# print("BEST K")
# print("=" * 60)

# print("Best K:", best_k)
# print(
#     "Best Accuracy:",
#     f"{accuracies[best_index]:.4f}"
# )


"""
Elevate Labs - AI & ML Internship
Task 6: K-Nearest Neighbors (KNN) Classification

Stage 4:
Model evaluation and visualization.

This stage includes:
- KNN training
- K-value experimentation
- Accuracy evaluation
- Classification report
- Confusion matrix
- K vs Accuracy visualization
- Decision boundary visualization
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ==============================================================
# PROJECT PATHS
# ==============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================
# 1. LOAD DATASET
# ==============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ELEVATE LABS - TASK 6")
print("K-NEAREST NEIGHBORS CLASSIFICATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDataset preview:")
print(df.head())

print("\nClass distribution:")
print(df["Species"].value_counts())


# ==============================================================
# 2. SELECT FEATURES AND TARGET
# ==============================================================

feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

X = df[feature_columns]
y = df["Species"]


# ==============================================================
# 3. TRAIN-TEST SPLIT
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================================================
# 4. FEATURE NORMALIZATION
# ==============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature normalization completed.")


# ==============================================================
# 5. EXPERIMENT WITH DIFFERENT K VALUES
# ==============================================================

k_values = range(1, 21)

accuracies = []

print("\n" + "=" * 70)
print("K VALUE EXPERIMENT")
print("=" * 70)

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


# ==============================================================
# 6. FIND BEST K
# ==============================================================

best_index = int(np.argmax(accuracies))

best_k = list(k_values)[best_index]

best_accuracy = accuracies[best_index]

print("\n" + "=" * 70)
print("BEST K")
print("=" * 70)

print("Best K:", best_k)

print(
    "Best Accuracy:",
    f"{best_accuracy:.4f}"
)


# ==============================================================
# 7. TRAIN FINAL KNN MODEL
# ==============================================================

final_model = KNeighborsClassifier(
    n_neighbors=best_k
)

final_model.fit(
    X_train_scaled,
    y_train
)


# ==============================================================
# 8. MAKE FINAL PREDICTIONS
# ==============================================================

final_predictions = final_model.predict(
    X_test_scaled
)


# ==============================================================
# 9. CALCULATE FINAL ACCURACY
# ==============================================================

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Final Test Accuracy: "
    f"{final_accuracy:.4f}"
)

print(
    f"Final Test Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ==============================================================
# 10. CLASSIFICATION REPORT
# ==============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        final_predictions
    )
)


# ==============================================================
# 11. CONFUSION MATRIX
# ==============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=labels
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# Create confusion matrix visualization

fig, ax = plt.subplots(
    figsize=(8, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    f"KNN Confusion Matrix (K={best_k})"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200
)

plt.close()


# ==============================================================
# 12. K VS ACCURACY VISUALIZATION
# ==============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    list(k_values),
    accuracies,
    marker="o"
)

plt.axvline(
    best_k,
    linestyle="--",
    label=f"Best K = {best_k}"
)

plt.scatter(
    [best_k],
    [best_accuracy],
    s=80,
    zorder=3
)

plt.xlabel(
    "Number of Neighbors (K)"
)

plt.ylabel(
    "Test Accuracy"
)

plt.title(
    "KNN: K Value vs Test Accuracy"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "k_vs_accuracy.png",
    dpi=200
)

plt.close()


# ==============================================================
# 13. DECISION BOUNDARY VISUALIZATION
# ==============================================================

"""
The final KNN model uses all four Iris features.

A four-dimensional decision boundary cannot be directly
visualized on a normal 2-D graph.

Therefore, Petal Length and Petal Width are used to create
a separate 2-D visualization of the KNN decision regions.
"""

boundary_features = [
    "PetalLengthCm",
    "PetalWidthCm",
]

X_2d = df[boundary_features]


# Split 2-D data

X2_train, _, y2_train, _ = train_test_split(
    X_2d,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Normalize 2-D features

scaler_2d = StandardScaler()

X2_train_scaled = scaler_2d.fit_transform(
    X2_train
)


# Train KNN for visualization

boundary_model = KNeighborsClassifier(
    n_neighbors=best_k
)

boundary_model.fit(
    X2_train_scaled,
    y2_train
)


# Create coordinate grid

x_min = X2_train_scaled[:, 0].min() - 1
x_max = X2_train_scaled[:, 0].max() + 1

y_min = X2_train_scaled[:, 1].min() - 1
y_max = X2_train_scaled[:, 1].max() + 1


xx, yy = np.meshgrid(
    np.linspace(
        x_min,
        x_max,
        500
    ),
    np.linspace(
        y_min,
        y_max,
        500
    )
)


# Predict every point in the grid

grid_predictions = boundary_model.predict(
    np.c_[
        xx.ravel(),
        yy.ravel()
    ]
)

grid_predictions = grid_predictions.reshape(
    xx.shape
)


# Convert class names to numbers

classes = sorted(
    y.unique()
)

class_to_number = {
    class_name: index
    for index, class_name in enumerate(classes)
}

Z = np.vectorize(
    class_to_number.get
)(
    grid_predictions
)


# Plot decision regions

plt.figure(
    figsize=(10, 7)
)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    levels=np.arange(
        len(classes) + 1
    ) - 0.5
)


# Plot training points

for class_name in classes:

    mask = y2_train == class_name

    plt.scatter(
        X2_train_scaled[mask, 0],
        X2_train_scaled[mask, 1],
        label=class_name,
        edgecolors="black",
        s=45
    )


plt.xlabel(
    "Standardized Petal Length"
)

plt.ylabel(
    "Standardized Petal Width"
)

plt.title(
    f"KNN Decision Boundary "
    f"using Petal Features (K={best_k})"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_boundary.png",
    dpi=200
)

plt.close()


# ==============================================================
# 14. SAVE K RESULTS
# ==============================================================

results = pd.DataFrame({
    "K": list(k_values),
    "Test_Accuracy": accuracies
})

results.to_csv(
    OUTPUT_DIR / "k_results.csv",
    index=False
)


# ==============================================================
# 15. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("TASK 6 STAGE 4 COMPLETED")
print("=" * 70)

print(
    f"Best K: {best_k}"
)

print(
    f"Final Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print("\nGenerated files:")

print(
    "1. outputs/k_vs_accuracy.png"
)

print(
    "2. outputs/confusion_matrix.png"
)

print(
    "3. outputs/decision_boundary.png"
)

print(
    "4. outputs/k_results.csv"
)

print("\nAll evaluation and visualization steps completed successfully.")