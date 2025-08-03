import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = ""
OUTPUT_DIR = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_energy_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sum(grad_x**2 + grad_y**2)
    return np.array([energy])

def extract_energy_from_images():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")])
    for fname in tqdm(files):
        img_path = os.path.join(INPUT_DIR, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        feature = compute_energy_feature(image)
        out_name = fname.replace(".jpg", ".txt")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        np.savetxt(out_path, feature, fmt="%.6f", delimiter=",")

    print(f"Energy feature extraction complete. Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_energy_from_images()
