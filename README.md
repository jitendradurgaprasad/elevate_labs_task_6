# Elevate Labs Task 6 - K-Nearest Neighbors Classification

## Objective

Implement and understand the K-Nearest Neighbors (KNN) algorithm for a classification problem using the Iris dataset.

The project demonstrates:

- Instance-based learning
- Feature normalization
- Euclidean distance based classification
- K-value selection
- Model evaluation
- Confusion matrix
- Decision boundary visualization

---

## Dataset

The project uses the Iris dataset.

The dataset contains:

- 150 samples
- 4 numerical features
- 3 target classes

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Classes

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The `Id` column is not used as a predictive feature because it is only an identifier.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure

```text
Elevate-Task-6/
│
├── data/
│   └── Iris.csv
│
├── src/
│   └── knn_classification.py
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── decision_boundary.png
│   ├── k_results.csv
│   └── k_vs_accuracy.png
│
├── README.md
└── requirements.txt