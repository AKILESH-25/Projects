import os
import cv2
import numpy as np

def load_images_from_folder(folder, img_size=(100, 100)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for identity in sorted(os.listdir(folder)):
        person_path = os.path.join(folder, identity)
        if not os.path.isdir(person_path):
            continue

        if identity not in label_map:
            label_map[identity] = label_counter
            label_counter += 1

        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                images.append(img_resized.flatten())  # flatten to 1D
                labels.append(label_map[identity])

    return np.array(images), np.array(labels), label_map
