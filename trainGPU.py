from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import glob
from collections import defaultdict
import shutil
import re
import torch


def check_device():
    """Kullanılabilir cihazları kontrol eder ve en uygun cihazı seçer"""
    print("\n=== CİHAZ BİLGİLERİ ===")
    print(f"CUDA kullanılabilir mi: {torch.cuda.is_available()}")
    print(f"MPS kullanılabilir mi: {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon GPU (MPS) kullanılıyor")
    else:
        device = 'cpu'
        print("CPU kullanılıyor")

    return device


def check_and_create_directories(base_dir):
    """Gerekli dizinleri oluşturur ve kontrol eder"""
    directories = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        Path(full_path).mkdir(parents=True, exist_ok=True)
        print(f"Dizin kontrol edildi/oluşturuldu: {full_path}")


def analyze_dataset(base_dir):
    """Veri setindeki sınıfları detaylı analiz eder"""
    label_dirs = ['labels/train', 'labels/val', 'labels/test']
    class_stats = defaultdict(lambda: {
        'count': 0,
        'files': set(),
        'names': set(),
        'splits': defaultdict(int)
    })

    name_pattern = re.compile(r'_([a-zA-Z]+)_')

    total_files = 0
    for label_dir in label_dirs:
        dir_path = os.path.join(base_dir, label_dir)
        if not os.path.exists(dir_path):
            continue

        split_name = label_dir.split('/')[-1]

        for label_file in glob.glob(os.path.join(dir_path, '*.txt')):
            total_files += 1
            try:
                filename = os.path.basename(label_file)
                name_match = name_pattern.search(filename)
                if name_match:
                    class_name = name_match.group(1).lower()

                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            stats = class_stats[class_id]
                            stats['count'] += 1
                            stats['files'].add(filename)
                            if name_match:
                                stats['names'].add(class_name)
                            stats['splits'][split_name] += 1

            except Exception as e:
                print(f"Hata - {label_file}: {str(e)}")

    print("\n=== VERI SETI SINIF ANALIZI ===")
    print(f"\nToplam dosya sayısı: {total_files}")
    print(f"Toplam benzersiz sınıf sayısı: {len(class_stats)}")

    print("\nSINIF DETAYLARI:")
    print("-" * 50)
    for class_id, stats in sorted(class_stats.items()):
        print(f"\nSınıf ID: {class_id}")
        print(f"Toplam örnek sayısı: {stats['count']}")
        print(f"Benzersiz dosya sayısı: {len(stats['files'])}")
        if stats['names']:
            print(f"Tespit edilen sınıf isimleri: {', '.join(sorted(stats['names']))}")
        print("\nVeri seti dağılımı:")
        for split, count in stats['splits'].items():
            print(f"- {split}: {count} örnek")
        print(f"Örnek dosyalar: {', '.join(list(stats['files'])[:3])}...")
        print("-" * 50)

    return class_stats


def create_class_mapping():
    """Sınıf eşleştirmelerini oluşturur"""
    # Ana sınıf isimleri
    class_names = ['banana', 'blackberries', 'raspberry', 'lemon',
                   'grapes', 'tomato', 'apple', 'chilli']

    # Eski ID'leri yeni ID'lere eşleştirme
    mapping = {
        # Muz (Banana)
        '0': '0',  # white background
        '1': '0',  # without background
        # Böğürtlen (Blackberries)
        '2': '1',
        # Ahududu (Raspberry)
        '3': '2',
        # Limon (Lemon)
        '4': '3',  # white background
        '5': '3',  # without background
        # Üzüm (Grapes)
        '6': '4',  # white background
        '7': '4',  # without background
        # Domates (Tomato)
        '8': '5',  # white background
        '9': '5',  # without background
        # Elma (Apple)
        '10': '6',  # white background
        '11': '6',  # without background
        # Biber (Chilli)
        '12': '7',  # white background
        '13': '7'  # without background
    }

    return mapping, class_names


def fix_labels(base_dir, class_mapping):
    """Etiket dosyalarını yeni mapping'e göre düzeltir"""
    label_dirs = ['labels/train', 'labels/val', 'labels/test']
    fixed_count = 0

    for label_dir in label_dirs:
        dir_path = os.path.join(base_dir, label_dir)
        if not os.path.exists(dir_path):
            continue

        for label_file in glob.glob(os.path.join(dir_path, '*.txt')):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                fixed_lines = []
                file_modified = False
                for line in lines:
                    parts = line.strip().split()
                    if parts and parts[0] in class_mapping:
                        parts[0] = class_mapping[parts[0]]
                        fixed_lines.append(' '.join(parts) + '\n')
                        fixed_count += 1
                        file_modified = True

                if file_modified:
                    # Yedek dosya oluştur
                    backup_file = label_file + '.backup'
                    if not os.path.exists(backup_file):
                        shutil.copy2(label_file, backup_file)

                    # Düzeltilmiş içeriği yaz
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)

            except Exception as e:
                print(f"Hata - {label_file}: {str(e)}")

    print(f"\nToplam {fixed_count} etiket düzeltildi.")


def create_dataset_config(base_dir, class_names):
    """Dataset yapılandırma dosyasını oluşturur"""
    data_yaml = {
        'path': base_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }

    config_path = os.path.join(base_dir, 'dataset.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"\nDataset yapılandırması oluşturuldu: {config_path}")
    print(f"Toplam sınıf sayısı: {len(class_names)}")
    print("Sınıf isimleri:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    return config_path


def train_yolov8(config_path, device):
    """YOLOv8 modelini eğitir"""
    try:
        model = YOLO('yolov8n.pt')  # YOLOv8n modelini yükle

        # Eğitim parametreleri
        results = model.train(
            data=config_path,
            epochs=50,  # Epoch sayısı
            imgsz=640,  # Görüntü boyutu
            batch=8,  # Batch size
            name='fruit_detection_final',
            device=device,  # Otomatik tespit edilen cihaz
            patience=20,  # Early stopping sabır değeri
            save=True,  # En iyi modeli kaydet
            verbose=True,  # Detaylı çıktı
            pretrained=True,  # Önceden eğitilmiş ağırlıkları kullan
            optimizer='AdamW',  # Optimizer seçimi
            lr0=0.001,  # Başlangıç learning rate
            weight_decay=0.0005,  # Weight decay
            # Veri arttırma parametreleri
            degrees=10.0,  # Rotasyon
            translate=0.1,  # Çeviri
            scale=0.5,  # Ölçekleme
            shear=2.0,  # Kesme
            flipud=0.0,  # Dikey çevirme
            fliplr=0.5,  # Yatay çevirme
            mosaic=1.0,  # Mozaik
            mixup=0.0  # Mixup
        )
        return results

    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {str(e)}")
        return None


def main():
    # Ana dizin
    base_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj'

    # 1. Cihaz kontrolü
    device = check_device()
    print(f"\nSeçilen cihaz: {device}")

    # 2. Dizinleri kontrol et ve oluştur
    print("\nDizinler kontrol ediliyor...")
    check_and_create_directories(base_dir)

    # 3. Mevcut sınıfları analiz et
    print("\nVeri seti analiz ediliyor...")
    class_stats = analyze_dataset(base_dir)

    # 4. Sınıf eşleştirmelerini oluştur
    print("\nSınıf eşleştirmeleri oluşturuluyor...")
    class_mapping, class_names = create_class_mapping()

    # 5. Etiketleri düzelt
    print("\nEtiketler düzeltiliyor...")
    fix_labels(base_dir, class_mapping)

    # 6. Dataset yapılandırmasını oluştur
    print("\nDataset yapılandırması oluşturuluyor...")
    config_path = create_dataset_config(base_dir, class_names)

    # 7. Son kontrol
    print("\nSon durum kontrol ediliyor...")
    final_stats = analyze_dataset(base_dir)

    # 8. Eğitimi başlat
    print("\nEğitim başlatılıyor...")
    results = train_yolov8(config_path, device)

    if results:
        print("\nEğitim başarıyla tamamlandı!")
        print(f"Eğitilmiş model ve sonuçlar: runs/detect/fruit_detection_final/")
    else:
        print("\nEğitim sırasında bir hata oluştu!")


if __name__ == "__main__":
    main()