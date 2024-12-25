from ultralytics import YOLO
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(weights_path, data_yaml):
    """
    Comprehensive model evaluation
    """
    model = YOLO(weights_path)

    # Run validation on test set
    results = model.val(data=data_yaml, split='test')

    # Get metrics
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'Speed (ms)': results.speed['inference']
    }

    # Class-wise performance
    class_map = results.box.ap_class_dict

    return metrics, class_map


def plot_class_performance(class_map):
    """
    Visualize per-class performance
    """
    classes = list(class_map.keys())
    ap_values = list(class_map.values())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=ap_values, y=classes)
    plt.title('Per-class Average Precision (AP)')
    plt.xlabel('AP')
    plt.tight_layout()
    plt.savefig('class_performance.png')
    plt.close()


def main():
    # Update these paths
    weights_path = 'runs/detect/fruit_detection/weights/best.pt'
    data_yaml = 'dataset.yaml'

    # Evaluate model
    metrics, class_map = evaluate_model(weights_path, data_yaml)

    # Print overall metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Print per-class performance
    print("\nPer-class Average Precision:")
    for class_name, ap in class_map.items():
        print(f"{class_name}: {ap:.4f}")

    # Visualize results
    plot_class_performance(class_map)


if __name__ == "__main__":
    main()