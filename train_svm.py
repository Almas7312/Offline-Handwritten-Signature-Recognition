import os
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -------------------------
# LOAD IMAGES
# -------------------------

def load_images(data_path, label):
    images = []
    labels = []

    for file_name in os.listdir(data_path):

        if file_name.endswith(".png") or file_name.endswith(".jpg"):

            img_path = os.path.join(data_path, file_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels


# -------------------------
# ENVELOPE EXTRACTION
# -------------------------

def calculate_envelopes(img):

    upper = np.max(img, axis=0)

    lower = np.min(img, axis=0)

    return upper, lower


# -------------------------
# FEATURE EXTRACTION
# -------------------------

def extract_statistical_features(images):

    features = []

    for img in images:

        upper, lower = calculate_envelopes(img)

        min_len = min(len(upper), len(lower))

        upper = upper[:min_len]
        lower = lower[:min_len]

        envelopes = np.concatenate([upper, lower])

        mean = np.mean(envelopes)

        std = np.std(envelopes)

        var = np.var(envelopes)

        skewness = np.mean(
            (envelopes - mean) ** 3
        ) / (std ** 3 if std > 0 else 1)

        kurtosis = np.mean(
            (envelopes - mean) ** 4
        ) / (std ** 4 if std > 0 else 1) - 3

        features.append(
            [
                mean,
                std,
                var,
                skewness,
                kurtosis
            ]
        )

    return np.array(features)


# -------------------------
# DATASET PATHS
# -------------------------

original_path = "Downloads/FDS/signatures/full_org"

forged_path = "Downloads/FDS/signatures/full_forg"

# -------------------------
# LOAD DATA
# -------------------------

original_images, original_labels = load_images(
    original_path,
    label=1
)

forged_images, forged_labels = load_images(
    forged_path,
    label=0
)

all_images = original_images + forged_images

all_labels = original_labels + forged_labels

# -------------------------
# FEATURE EXTRACTION
# -------------------------

features = extract_statistical_features(all_images)

print("Feature Shape:", features.shape)

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    all_labels,
    test_size=0.3,
    random_state=42
)

# -------------------------
# STANDARDIZATION
# -------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# -------------------------
# PCA
# -------------------------

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)

# -------------------------
# SVM TRAINING
# -------------------------

param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(class_weight='balanced'),
    param_grid,
    cv=5
)

grid_search.fit(
    X_train_pca,
    y_train
)

best_model = grid_search.best_estimator_

# -------------------------
# PREDICTION
# -------------------------

y_pred = best_model.predict(
    X_test_pca
)

# -------------------------
# RESULTS
# -------------------------

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
