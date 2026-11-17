# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:28:03 2026

@author: unangsunarya
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt




# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "dataset/iris.csv"

df = pd.read_csv(file_path)

print("=== Iris Dataset ===")
print(df.head())

print("\n=== Dataset Information ===")
print(df.info())

print("\n=== Jumlah Data ===")
print(df.shape)


# ============================================================
# 2. MEMISAHKAN FEATURES DAN TARGET
# ============================================================

X = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values

X=np.array(X)
y=np.array(y)

print("\nFeatures:")
print(X[:5])

print("\nTarget:")
print(y[:5])


# ============================================================
# 3. ENCODING TARGET
# ============================================================

# Mengubah nama kelas menjadi angka
classes = np.unique(y)

class_mapping = {
    class_name: index
    for index, class_name in enumerate(classes)
}

y_encoded = np.array([
    class_mapping[label]
    for label in y
])

print("\nClass Mapping:")
print(class_mapping)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.arange(len(X))

# Mengacak index
np.random.shuffle(indices)

# 80% training, 20% testing
train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]

print("\nTraining data :", X_train.shape)
print("Testing data  :", X_test.shape)


# ============================================================
# 5. NORMALISASI DATA
# ============================================================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


# ============================================================
# 6. FUNGSI EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# ============================================================
# 7. K-NEAREST NEIGHBORS
# ============================================================

def knn_predict(X_train, y_train, X_test, k=5):

    predictions = []

    for test_sample in X_test:

        distances = []

        for i, train_sample in enumerate(X_train):

            distance = euclidean_distance(
                test_sample,
                train_sample
            )

            distances.append((distance, y_train[i]))

        # Urutkan berdasarkan distance
        distances.sort(key=lambda x: x[0])

        # Ambil k tetangga terdekat
        k_neighbors = distances[:k]

        # Ambil label
        labels = [label for distance, label in k_neighbors]

        # Majority voting
        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        prediction = unique_labels[
            np.argmax(counts)
        ]

        predictions.append(prediction)

    return np.array(predictions)


# ============================================================
# 8. TRAINING / PREDICTION
# ============================================================

k = 5

y_pred = knn_predict(
    X_train_scaled,
    y_train,
    X_test_scaled,
    k
)


# ============================================================
# 9. EVALUASI AKURASI
# ============================================================

accuracy = np.mean(y_pred == y_test)

print("\n=== HASIL KLASIFIKASI ===")
print("K =", k)
print("Accuracy =", accuracy)
print("Accuracy (%) =", accuracy * 100)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

num_classes = len(classes)

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for actual, predicted in zip(y_test, y_pred):

    confusion_matrix[actual, predicted] += 1


print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix)


# ============================================================
# 11. VISUALISASI CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    confusion_matrix,
    cmap="Blues"
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(num_classes),
    classes
)

plt.yticks(
    range(num_classes),
    classes
)

# Menampilkan angka pada matrix
for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()


# ============================================================
# 12. VISUALISASI DATA IRIS
# ============================================================

plt.figure(figsize=(8, 6))

for class_index, class_name in enumerate(classes):

    data = X[y_encoded == class_index]

    plt.scatter(
        data[:, 0],
        data[:, 1],
        label=class_name
    )

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.title("Iris Dataset")
plt.legend()

plt.grid(True)
plt.show()


