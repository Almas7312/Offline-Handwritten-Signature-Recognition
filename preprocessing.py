import os
import cv2

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
