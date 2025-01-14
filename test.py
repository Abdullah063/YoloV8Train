from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np


def check_device():
    """Kullanılabilir cihazları kontrol eder ve en uygun cihazı seçer"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon GPU kullanılıyor")
    else:
        device = 'cpu'
        print("CPU kullanılıyor")
    return device


class FruitDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Args:
            model_path: Eğitilmiş model dosyasının yolu
            conf_threshold: Güven eşik değeri
        """
        self.device = check_device()
        self.conf_threshold = conf_threshold

        # Modeli yükle
        self.model = YOLO(model_path)
        print(f"Model başarıyla yüklendi: {model_path}")

    def process_frame(self, frame):
        """Tek bir frame'i işler ve sonuçları döndürür"""
        results = self.model(frame, conf=self.conf_threshold)[0]
        annotated_frame = results.plot()

        # FPS hesapla
        fps = 1.0 / results.speed['inference']

        # FPS'i ekrana yaz
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame, results

    def start_webcam(self):
        """Webcam'den gerçek zamanlı tespit başlatır"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Webcam açılamadı!")
            return

        print("Webcam başlatıldı. Çıkmak için 'q' tuşuna basın.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, results = self.process_frame(frame)

                # Sonuçları göster
                cv2.imshow("Fruit Detection", annotated_frame)

                # 'q' tuşuna basılırsa çık
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_video(self, video_path, output_path=None):
        """Video dosyasını işler"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Video açılamadı: {video_path}")
            return

        # Video özellikleri
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Çıktı video yazıcısı
        if output_path:
            writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"Frame işleniyor: {frame_count}/{total_frames}", end='\r')

                annotated_frame, results = self.process_frame(frame)

                if output_path:
                    writer.write(annotated_frame)

                # Sonuçları göster
                cv2.imshow("Fruit Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if output_path:
                writer.release()
            cv2.destroyAllWindows()
            print("\nVideo işleme tamamlandı!")


def main():
    # Eğitilmiş model yolu - bu yolu kendi modelinizin yoluna göre değiştirin
    model_path = "runs/detect/fruit_detection_final/weights/best.pt"

    # Dedektör oluştur
    detector = FruitDetector(model_path, conf_threshold=0.5)

    # Kullanım modu seçimi
    print("\nKullanım modu seçin:")
    print("1: Webcam")
    print("2: Video dosyası")
    choice = input("Seçiminiz (1/2): ")

    if choice == "1":
        detector.start_webcam()
    elif choice == "2":
        video_path = input("Video dosyası yolunu girin: ")
        output_path = input("Çıktı video yolunu girin (boş bırakılabilir): ").strip()
        output_path = output_path if output_path else None
        detector.process_video(video_path, output_path)
    else:
        print("Geçersiz seçim!")


if __name__ == "__main__":
    main()