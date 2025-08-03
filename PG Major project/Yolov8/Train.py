from ultralytics import YOLO

def main():
    # Load model (base or custom if available)
    model = YOLO("yolov8s.pt")  
    # Train the model
    model.train(
        data="yolov8/config/visdrone.yaml",
        cfg="yolov8/config/msfe_yolo.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="msfe_yolo_training"
    )

    # Evaluate the model
    metrics = model.val()
    print("Validation results:", metrics)

    # Export the trained model
    model.export(format="onnx")  # or "torchscript" or "engine" for deployment

if __name__ == "__main__":
    main()
