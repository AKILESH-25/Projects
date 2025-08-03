import os
import cv2
from tqdm import tqdm

# Configurable Paths
INPUT_DIR = "../dataset/all_images"
OUTPUT_DIR = "../dataset/resized_images"

# Resize parameters
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640

# Normalization flag (optional)
NORMALIZE = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path, normalize=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    resized_img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    if normalize:
        resized_img = resized_img / 255.0  # convert to float, range [0,1]

    return resized_img

def process_all_images():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))])
    print(f"🔄 Processing {len(image_files)} images...")

    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        processed_img = preprocess_image(img_path, normalize=NORMALIZE)

        if processed_img is not None:
            out_path = os.path.join(OUTPUT_DIR, filename)

            # Save normalized images back to [0,255] if needed
            if NORMALIZE:
                processed_img = (processed_img * 255).astype('uint8')

            cv2.imwrite(out_path, processed_img)

    print(f"Preprocessing complete. Resized images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_images()
