import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# -------------------------
# CONFUSION MATRIX
# -------------------------

def plot_confusion_matrix(
    y_test,
    y_pred
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Forged",
            "Original"
        ],
        yticklabels=[
            "Forged",
            "Original"
        ]
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.show()


# -------------------------
# PRECISION RECALL F1
# -------------------------

def plot_metrics(
    y_test,
    y_pred
):

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    metrics = [
        "precision",
        "recall",
        "f1-score"
    ]

    forged = [
        report["0"][m]
        for m in metrics
    ]

    genuine = [
        report["1"][m]
        for m in metrics
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars1 = ax.bar(
        x - width/2,
        forged,
        width,
        label="Forged"
    )

    bars2 = ax.bar(
        x + width/2,
        genuine,
        width,
        label="Original"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        metrics
    )

    ax.legend()

    plt.show()


# -------------------------
# PCA VISUALIZATION
# -------------------------

def plot_feature_space(
    X_train_pca,
    X_test_pca
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        X_train_pca[:,0],
        X_train_pca[:,1],
        label="Train"
    )

    plt.scatter(
        X_test_pca[:,0],
        X_test_pca[:,1],
        label="Test",
        marker="x"
    )

    plt.title(
        "Signature Feature Space"
    )

    plt.xlabel(
        "PCA Component 1"
    )

    plt.ylabel(
        "PCA Component 2"
    )

    plt.legend()

    plt.show()


# -------------------------
# GENUINE VS FORGED
# -------------------------

def compare_signatures(
    genuine_path,
    forged_path
):

    genuine = cv2.imread(
        genuine_path,
        cv2.IMREAD_GRAYSCALE
    )

    forged = cv2.imread(
        forged_path,
        cv2.IMREAD_GRAYSCALE
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10,5)
    )

    ax[0].imshow(
        genuine,
        cmap="gray"
    )

    ax[0].set_title(
        "Genuine"
    )

    ax[0].axis("off")

    ax[1].imshow(
        forged,
        cmap="gray"
    )

    ax[1].set_title(
        "Forged"
    )

    ax[1].axis("off")

    plt.show()
