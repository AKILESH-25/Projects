import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
IMAGE_DIR = ""
LABEL_DIR = ""
OUTPUT_DIR = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_roi(image, label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    height, width = image.shape[:2]
    roi_images = []

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x_center, y_center, w, h = map(float, parts)
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            roi_images.append((i, roi))

    return roi_images

def process_images():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
    for img_file in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, img_file.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(label_path):
            continue

        rois = extract_roi(image, label_path)

        for i, roi in rois:
            out_name = f"{os.path.splitext(img_file)[0]}_roi{i}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            resized_roi = cv2.resize(roi, (64, 64))
            cv2.imwrite(out_path, resized_roi)

    print(f"ROI extraction complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()
