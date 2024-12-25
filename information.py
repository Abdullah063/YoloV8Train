import os
import glob
from collections import defaultdict
import re


def analyze_dataset(base_dir):
    """Veri setindeki sınıfları detaylı analiz eder"""
    label_dirs = ['labels/train', 'labels/val', 'labels/test']
    class_stats = defaultdict(lambda: {
        'count': 0,
        'files': set(),
        'names': set(),
        'splits': defaultdict(int)
    })

    # Dosya isimlerinden sınıf isimlerini çıkarmak için regex
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
                # Dosya isminden sınıf ismini çıkar
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


if __name__ == "__main__":
    base_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj'
    analyze_dataset(base_dir)