import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

# --- Configuration ---
IMAGE_DIR = ""  
OUTPUT_DIR = ""
FEATURE_TYPE = "hog" 

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Feature Functions ---

def extract_hog_features(image):
    """
    Extract HOG features from a grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    return features


def extract_energy_features(image):
    """
    Compute energy feature: sum of squared gradients.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sum(grad_x**2 + grad_y**2)
    return np.array([energy])


# --- Main Feature Extraction Loop ---

def process_all_images(feature_type=FEATURE_TYPE):
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))
    ])

    print(f"Processing {len(image_files)} images using {feature_type.upper()} features...")

    for fname in tqdm(image_files):
        path = os.path.join(IMAGE_DIR, fname)
        image = cv2.imread(path)

        if image is None:
            print(f"[Warning] Unable to read image: {fname}")
            continue

        if feature_type == "hog":
            features = extract_hog_features(image)
        elif feature_type == "energy":
            features = extract_energy_features(image)
        else:
            raise ValueError("Unsupported feature type.")

        # Save features to a TXT file
        out_name = os.path.splitext(fname)[0] + ".txt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        np.savetxt(out_path, features, delimiter=",")
    
    print(f"Feature extraction complete. Features saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all_images()
