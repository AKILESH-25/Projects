import os
from preprocess import load_images_from_folder
from pca_module import PCAFeatureExtractor
from ann_model import train_and_evaluate
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/"

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    os.makedirs("results/evaluation", exist_ok=True)
    plt.savefig("results/evaluation/accuracy_plot.png")
    plt.close()

def main():
    print("🔍 Loading images...")
    X, y, label_map = load_images_from_folder(DATASET_PATH)
    print(f"Loaded {X.shape[0]} samples with {len(label_map)} classes")

    print("🔄 Applying PCA...")
    pca = PCAFeatureExtractor(n_components=100)
    X_pca = pca.fit_transform(X)

    print("🧠 Training ANN model...")
    model, history = train_and_evaluate(X_pca, y)

    print("📊 Saving accuracy plot...")
    plot_accuracy(history)

    print("✅ Done!")

if __name__ == "__main__":
    main()
