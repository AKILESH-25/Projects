# 🚗 Graph-Based Optimization for Multi-Scale Vehicle Detection and Recognition

This project implements a graph-based object optimization technique combined with machine learning to detect and classify vehicles of multiple scales from video footage. Developed as part of a PG mini project, it focuses on detecting vehicles from aerial or ground-view surveillance footage with variable resolutions and complex backgrounds. The system includes preprocessing, object shape modeling, feature extraction, and vehicle classification using an Artificial Neural Network (ANN).

---

## 📌 Features

- 🎞️ Frame extraction from surveillance or traffic video
- 📉 Background subtraction and median filtering to remove noise
- 🔳 ROI extraction using label maps and bounding box parsing
- 🔁 Graph-based shape optimization for precise object contour modeling
- ⚙️ Feature extraction:
  - Histogram of Oriented Gradients (HOG)
  - Energy-based features (spatial texture/gradient descriptors)
- 🧠 Vehicle classification using a trained Artificial Neural Network (ANN)
- 📈 Model evaluation with accuracy, loss curves, and confusion matrix
- 💾 Feature and label export to CSV and TXT formats for training

---

## 📂 Folder Structure

