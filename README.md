# Offline-Handwritten-Signature-Recognition


This project presents an offline handwritten signature recognition and verification system based on upper and lower envelope extraction, Eigenvalue analysis, Principal Component Analysis (PCA), and Support Vector Machine (SVM) classification.

The system extracts structural characteristics from handwritten signatures by identifying upper and lower envelopes and computing statistical and Eigenvalue-based features. PCA is applied for dimensionality reduction, and a linear SVM classifier is trained to distinguish genuine signatures from forged signatures.

## Key Features

* Signature preprocessing and normalization
* Upper and lower envelope extraction
* Eigenvalue-based feature extraction
* PCA-based dimensionality reduction
* Linear SVM classification
* Genuine vs Forged signature verification

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* PCA
* SVM
* Matplotlib

## Results

The proposed system achieved an accuracy of approximately 71% in distinguishing genuine and forged signatures while maintaining efficient feature representation through PCA.

## Future Work

* Deep Learning-based signature verification
* CNN and Transformer architectures
* Larger benchmark datasets
* Improved forgery detection techniques
