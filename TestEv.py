from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
import torch
import json
from pathlib import Path
import pandas as pd


class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        self.model = YOLO(model_path)
        self.test_data = test_data_path
        self.class_names = self.model.names

    def evaluate_model(self):
        print("\nModel değerlendirmesi başlatılıyor...")
        results = self.model.val(data=self.test_data, verbose=True)
        return results

    def plot_confusion_matrix(self, results):
        plt.figure(figsize=(12, 8))
        confusion_matrix = results.confusion_matrix.matrix

        # Normalize confusion matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_conf_mx = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            norm_conf_mx = np.nan_to_num(norm_conf_mx)

        sns.heatmap(norm_conf_mx, annot=True, fmt='.2f',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap='Blues')

        plt.title('Normalize Edilmiş Karışıklık Matrisi')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall + 1e-16)

    def generate_detailed_report(self, results):
        # Genel metrikler alınıyor
        mean_results = results.mean_results()

        # Sınıf sayısını al
        num_classes = len(self.class_names)

        report = {
            "Genel Metrikler": {
                "mAP50": float(np.mean(results.maps[0])),  # mAP50 ortalaması
                "mAP50-95": float(results.maps[1]),  # mAP50-95
                "Precision": float(mean_results[0]),  # precision
                "Recall": float(mean_results[1]),  # recall
            }
        }

        # F1-Score hesapla
        report["Genel Metrikler"]["F1-Score"] = self.calculate_f1_score(
            report["Genel Metrikler"]["Precision"],
            report["Genel Metrikler"]["Recall"]
        )

        # Sınıf bazında metrikler
        report["Sınıf Bazında Metrikler"] = {}

        # results_dict'ten metrikleri al
        results_dict = results.results_dict

        for i, class_name in enumerate(self.class_names):
            try:
                # Her sınıf için metrikleri al
                precision = float(results_dict['metrics/precision(B)'][i])
                recall = float(results_dict['metrics/recall(B)'][i])
                map50 = float(results.maps[0][i])

                class_metrics = {
                    "Precision": precision,
                    "Recall": recall,
                    "mAP50": map50,
                    "F1-Score": self.calculate_f1_score(precision, recall)
                }

                report["Sınıf Bazında Metrikler"][class_name] = class_metrics

            except Exception as e:
                print(f"Uyarı: {class_name} sınıfı için metrik hesaplaması başarısız: {str(e)}")
                continue

        # Raporu JSON olarak kaydet
        with open('evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        return report

    def print_report(self, report):
        print("\n=== MODEL DEĞERLENDİRME RAPORU ===")

        # Genel metrikler
        print("\nGenel Metrikler:")
        general_metrics = [[k, f"{v:.4f}"] for k, v in report["Genel Metrikler"].items()]
        print(tabulate(general_metrics, headers=["Metrik", "Değer"], tablefmt="grid"))

        # Sınıf bazında metrikler
        print("\nSınıf Bazında Metrikler:")
        class_data = []
        metrics_order = ["Precision", "Recall", "F1-Score", "mAP50"]

        for class_name, metrics in report["Sınıf Bazında Metrikler"].items():
            row = [class_name] + [f"{metrics[m]:.4f}" for m in metrics_order]
            class_data.append(row)

        headers = ["Sınıf"] + metrics_order
        print(tabulate(class_data, headers=headers, tablefmt="grid"))


def main():
    # Model ve test veri seti yolları
    model_path = "runs/detect/fruit_detection_final/weights/best.pt"
    test_data_path = "dataset.yaml"

    try:
        # Değerlendirici oluştur
        evaluator = ModelEvaluator(model_path, test_data_path)

        # Modeli değerlendir
        results = evaluator.evaluate_model()

        # Karışıklık matrisini çiz
        evaluator.plot_confusion_matrix(results)
        print("\nKarışıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")

        # Detaylı rapor oluştur
        report = evaluator.generate_detailed_report(results)
        print("Detaylı rapor 'evaluation_report.json' olarak kaydedildi.")

        # Raporu göster
        evaluator.print_report(report)

    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()