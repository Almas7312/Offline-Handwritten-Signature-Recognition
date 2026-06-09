import numpy as np

def calculate_envelopes(img):
    upper = np.max(img, axis=0)
    lower = np.min(img, axis=0)
    return upper, lower

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

        skewness = np.mean((envelopes - mean)**3) / (std**3 if std > 0 else 1)

        kurtosis = np.mean((envelopes - mean)**4) / (std**4 if std > 0 else 1) - 3

        features.append([mean, std, var, skewness, kurtosis])

    return np.array(features)
