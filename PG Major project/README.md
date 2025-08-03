# 🚁 SkyEye: Smarter Object Detection for UAVs

SkyEye is a deep learning–based object detection project designed to identify small and occluded objects from aerial drone footage. Inspired by the MSFE-YOLO architecture, this project improves upon YOLOv8 by incorporating multiscale feature extraction modules like SCF (Symmetric C2f), EMA (Efficient Multiscale Attention), and FF (Feature Fusion) to enhance accuracy in dense urban and low-resolution environments.

This project is tailored for real-time deployment on UAV platforms for surveillance, traffic monitoring, and disaster response.

---

## 🌐 Key Highlights

- 🔍 MSFE-YOLO architecture inspired by YOLOv8 with feature enhancement modules:
  - 🧱 SCF (Symmetric C2f): captures multiscale spatial features
  - 🧠 EMA (Efficient Multiscale Attention): emphasizes important contextual regions
  - 🔗 FF (Feature Fusion): merges high- and low-level features for refined detection
- 📷 Trained and evaluated on the VisDrone 2019 dataset
- 🎯 Capable of detecting small, distant, and occluded objects
- ⚙️ Fine-tuned using PyTorch and Ultralytics YOLOv8 implementation
- 📈 Achieved mAP@0.5 of 41.4% and mAP@0.5:0.95 of 25.2% on VisDrone test set

---

## 📁 Project Structure

