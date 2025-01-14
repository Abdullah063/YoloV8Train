import os
import glob
from collections import defaultdict
from pathlib import Path


def summarize_dataset(base_dir):
    """
    Veri seti istatistiklerini özetler
    Args:
        base_dir: Veri setinin ana dizini
    """
    stats = {
        'images': defaultdict(int),
        'labels': defaultdict(int),
        'classes': set(),
        'class_distribution': defaultdict(lambda: defaultdict(int))
    }

    # Her split için istatistikleri topla
    for split in ['train', 'val', 'test']:
        # Görüntü sayıları
        img_path = os.path.join(base_dir, split, 'images')
        if os.path.exists(img_path):
            image_files = glob.glob(os.path.join(img_path, '*.*'))
            stats['images'][split] = len([f for f in image_files
                                          if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png']])

        # Etiket sayıları ve sınıf dağılımı
        label_path = os.path.join(base_dir, split, 'labels')
        if os.path.exists(label_path):
            label_files = glob.glob(os.path.join(label_path, '*.txt'))
            stats['labels'][split] = len(label_files)

            # Her bir etiket dosyasını oku ve sınıfları say
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            class_id = int(line.strip().split()[0])
                            stats['classes'].add(class_id)
                            stats['class_distribution'][split][class_id] += 1
                except Exception as e:
                    print(f"Hata - {label_file}: {str(e)}")

    # YAML dosyasını kontrol et
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    yaml_status = "✅ Mevcut" if os.path.exists(yaml_path) else "❌ Eksik"

    # Sonuçları yazdır
    print("\n=== VERİ SETİ ÖZETİ ===")
    print(f"\ndataset.yaml durumu: {yaml_status}")

    print("\nGörüntü Sayıları:")
    print("-" * 30)
    total_images = 0
    for split, count in stats['images'].items():
        print(f"{split:>5}: {count:>5} görüntü")
        total_images += count
    print(f"TOPLAM: {total_images} görüntü")

    print("\nEtiket Sayıları:")
    print("-" * 30)
    total_labels = 0
    for split, count in stats['labels'].items():
        print(f"{split:>5}: {count:>5} etiket")
        total_labels += count
    print(f"TOPLAM: {total_labels} etiket")

    print("\nBenzersiz Sınıf Sayısı:", len(stats['classes']))
    print("Sınıf ID'leri:", sorted(list(stats['classes'])))

    print("\nSınıf Dağılımı:")
    print("-" * 30)
    for split in ['train', 'val', 'test']:
        if stats['class_distribution'][split]:
            print(f"\n{split} seti sınıf dağılımı:")
            for class_id in sorted(stats['class_distribution'][split].keys()):
                count = stats['class_distribution'][split][class_id]
                print(f"  Sınıf {class_id}: {count} örnek")

    # Uyarılar
    if total_images != total_labels:
        print("\n⚠️ UYARI: Görüntü ve etiket sayıları eşleşmiyor!")
        for split in ['train', 'val', 'test']:
            if stats['images'][split] != stats['labels'][split]:
                print(f"  - {split} setinde: {stats['images'][split]} görüntü, {stats['labels'][split]} etiket")


if __name__ == "__main__":
    base_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj'  # Kendi dizin yolunuzu buraya yazın
    summarize_dataset(base_dir)